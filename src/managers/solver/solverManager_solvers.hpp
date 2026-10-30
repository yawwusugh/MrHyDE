/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::steadySolver(vector<vector_RCP> & sol) {
  
  debugger->print("**** Starting SolverManager::steadySolver ...");
  
  for (int ss=0; ss<subcycles; ++ss) {
    for (size_t set=0; set<setnames.size(); ++set) {
      if (initial_type == "previous set") {
        this->setInitial(sol, set);
      }
      assembler->updatePhysicsSet(set);
      vector<vector_RCP> zero_soln;
      if (usestrongDBCs) {
        this->setDirichlet(set, sol[set]);
      }
      params->updateDynamicParams(0);
      this->nonlinearSolver(set, 0, sol, zero_soln, zero_soln,
                            zero_soln, zero_soln, zero_soln);
    }
  }
  postproc->record(sol, current_time, 0);
  
  debugger->print("**** Finished SolverManager::steadySolver");
  
}

// ========================================================================================
/* solve a transient problem */
// ========================================================================================

template<class Node>
void SolverManager<Node>::transientSolver(vector<vector_RCP> & initial, 
                                          MrHyDE_OptVector & gradient,
                                          ScalarT & start_time, ScalarT & end_time) {
  
  Teuchos::TimeMonitor localtimer(*transientsolvertimer);
  
  debugger->print(1, "******** Starting SolverManager::transientSolver ...");
  
  vector<vector_RCP> zero_vec(initial.size());
  
  current_time = start_time;
  if (!is_adjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector<vector_RCP> sol = initial;
    
    if (usestrongDBCs) {
      for (size_t set=0; set<initial.size(); ++set) {
        assembler->updatePhysicsSet(set);
        this->setDirichlet(set,sol[set]);
      }
    }
    
    postproc->record(sol,current_time,0);
    
    vector<vector<vector_RCP> > sol_prev;
    for (size_t set=0; set<sol.size(); ++set) {
      vector<vector_RCP> c_prev;
      for (int step=0; step<maxnumsteps[set]; ++step) {
        c_prev.push_back(linalg->getNewOverlappedVector(set));
      }
      sol_prev.push_back(c_prev);
    }
           
    int stepProg = 0;
    int numCuts = 0;
    int maxCuts = maxTimeStepCuts; // TMW: make this a user-defined input
    double timetol = end_time*1.0e-6; // just need to get close enough to final time
    
    while (current_time < (end_time-timetol) && numCuts<=maxCuts) {
      int status = 0;
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Time Step " << stepProg+1 << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      params->updateDynamicParams(stepProg);
      assembler->updateTimeStep(stepProg);
      
      //for (int ss=0; ss<subcycles; ++ss) {
        for (size_t set=0; set<sol.size(); ++set) {
          // this needs to come first now, so that updatePhysicsSet can pick out the
          // time integration info
          if (BDForder[set] > 1 && stepProg == startupSteps[set]) {
            // Only overwrite the current set
            this->setBackwardDifference(BDForder,set);
            this->setButcherTableau(ButcherTab,set);
          }

          assembler->updatePhysicsSet(set);
    
          // if num_stages = 1, the sol_stage = sol
          //
          vector<vector_RCP> sol_stage;
          if (maxnumstages[set] == 1) {
            sol_stage.push_back(sol[set]);
          }
          else {
            for (int stage=0; stage<maxnumstages[set]; ++stage) {
              sol_stage.push_back(linalg->getNewOverlappedVector(set));
              sol_stage[stage]->assign(*(sol[set]));
            }
          }
    
          // Increment the previous step solutions (shift history and moves u into first spot)
          for (size_t step=1; step<sol_prev[set].size(); ++step) {
            size_t ind = sol_prev[set].size()-step;
            sol_prev[set][ind]->assign(*(sol_prev[set][ind-1]));
          }
          sol_prev[set][0]->assign(*(sol[set]));
      
          ////////////////////////////////////////////////////////////////////////
          // Allow the groups to change subgrid model
          ////////////////////////////////////////////////////////////////////////
          
          vector<vector<int> > sgmodels = assembler->identifySubgridModels();
          multiscale_manager->update(sgmodels);
          
          for (int stage=0; stage<numstages[set]; stage++) {
            // Need a stage solution
            // Set the initial guess for stage solution
            // sol_stage[stage]->assign(*(sol_prev[0]));
            // Updates the current time and sets the stage number in wksets
            assembler->updateStage(stage, current_time, deltat); 

            if (usestrongDBCs) {
              this->setDirichlet(set, sol_stage[stage]);
            }
  
            if (fully_explicit) {
              status += this->explicitSolver(set, stage, sol, sol_stage, sol_prev[set], 
                                             zero_vec, zero_vec, zero_vec);
            }
            else {
              status += this->nonlinearSolver(set, stage, sol, sol_stage, sol_prev[set],
                                              zero_vec, zero_vec, zero_vec);
            }

            // u_{n+1} = u_n + \sum_stage ( u_stage - u_n )
            
            // if num_stages = 1, then we might be able to skip this 
            if (maxnumstages[set] > 1) {
              sol[set]->update(1.0, *(sol_stage[stage]), 1.0);
              sol[set]->update(-1.0, *(sol_prev[set][0]), 1.0);
            }
            multiscale_manager->completeStage();
          }
        }
      //}
      
      if (status == 0) { // NL solver converged
        current_time += deltat;
        stepProg += 1;
        multiscale_manager->completeTimeStep();
        postproc->record(sol,current_time,stepProg);
        
      }
      else { // something went wrong, cut time step and try again
        deltat *= 0.5;
        numCuts += 1;
        for (size_t set=0; set<sol.size(); ++set) {
          assembler->revertSoln(set);
          sol[set]->assign(*(sol_prev[set][0]));
        }
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << endl << endl << "*******************************************************" << endl;
          cout << endl << "**** Cutting time step to " << deltat << endl;
          cout << "**** Current time is " << current_time << endl << endl;
          cout << "*******************************************************" << endl << endl << endl;
        }
        
      }
    }
    // If the final step doesn't fall when a write is requested, catch that here  
    if (stepProg % postproc->write_frequency != 0 && postproc->write_solution) {
      postproc->writeSolution(sol, current_time);
    }
  }
  else if (fully_explicit) { // adjoint of the fully explicit E/B split (no AD)
    current_time = final_time;
    is_final_time = true;
    this->explicitSplitAdjoint(gradient);
  }
  else { // adjoint solve - fixed time stepping based on forward solve

    current_time = final_time;
    is_final_time = true;
    
    vector<vector_RCP> sol, sol_prev, phi, phi_prev, sol_stage, phi_stage;
    for (size_t set=0; set<1; ++set) { // hard coded for now
      sol.push_back(linalg->getNewOverlappedVector(set));
      sol_prev.push_back(linalg->getNewOverlappedVector(set));
      phi.push_back(linalg->getNewOverlappedVector(set));
      phi_prev.push_back(linalg->getNewOverlappedVector(set));
      sol_stage.push_back(linalg->getNewOverlappedVector(set));
      phi_stage.push_back(linalg->getNewOverlappedVector(set));
    }
    // Transient adjoints require derivatives of Jacobians w.r.t. previous states
    // We store the Jacobian-vector products in a (Nstep x Nstep) matrix
    if (previous_adjoints.size() == 0) {
      for (size_t i=0; i<numsteps[0]; ++i) { // hard-coded for now
        vector<vector_RCP> ivecs;
        for (size_t set=0; set<setnames.size(); ++set) { // hard-coded for now
          vector_RCP tempvec = linalg->getNewVector(set);
          tempvec->putScalar(0.0);
          ivecs.push_back(tempvec);
        }
        previous_adjoints.push_back(ivecs);
      }
    }
    else {
      for (size_t i=0; i<numsteps[0]; ++i) { // hard-coded for now
        for (size_t set=0; set<setnames.size(); ++set) { // hard-coded for now
          previous_adjoints[i][set]->putScalar(0.0);
        }
      }
    }
    
    size_t set = 0, stage = 0;
    // Just getting the number of times from first physics set should be fine
    // TODO will this be affected by having physics sets with different timesteppers?
    int store_index = 0;
    auto & soln_source = postproc->is_incremental_adjoint ? postproc->incr_soln : postproc->soln;
    size_t numFwdSteps = soln_source[set]->getTotalTimes(store_index)-1;

    for (size_t timeiter = 0; timeiter<numFwdSteps; timeiter++) {
      size_t cindex = numFwdSteps-timeiter;
      phi_prev[set] = linalg->getNewOverlappedVector(set);
      phi_prev[set]->update(1.0,*(phi[0]),0.0);
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Adjoint Time Step " << timeiter << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }

      // TMW: this is specific to implicit Euler
      // Needs to be generalized
      // Also, need to implement checkpoint/recovery
      bool fndu = soln_source[set]->extract(sol[set], cindex);
      if (!fndu) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to find forward solution");
      }
      bool fndup = soln_source[set]->extract(sol_prev[set], cindex-1);
      if (!fndup) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to find previous forward solution");
      }
      //params->updateDynamicParams(cindex-1);
      params->updateDynamicParams(cindex-1);
      //assembler->performGather(set,u_prev[set],0,0);
      //assembler->resetPrevSoln(set);

      int stime_index = cindex-1;

      current_time = soln_source[set]->getSpecificTime(store_index, stime_index);
      postproc->setTimeIndex(cindex);
      assembler->updateStage(stage, current_time, deltat);
      
      sol_stage[set]->assign(*sol[set]);
      // if multistage, recover forward solution at each stage
      if (numstages[set] == 1) { // No need to re-solve in this case
        int status = this->nonlinearSolver(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev);
        if (status>0) {
          // throw error
        }
        phi[set]->update(1.0,*(phi_stage[0]),0.0);
        postproc->computeSensitivities(sol, sol_stage, sol_prev, phi, current_time, cindex, deltat, gradient);
      }
      else {
        // NEEDS TO BE REWRITTEN
      }
      
      is_final_time = false;
      
    }
    
  }
  
  debugger->print(1, "******** Finished SolverManager::transientSolver");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::nonlinearSolver(const size_t & set, const size_t & stage,
                                         vector<vector_RCP> & sol, // [set]
                                         vector<vector_RCP> & sol_stage, // [stage]
                                         vector<vector_RCP> & sol_prev, // [step]
                                         vector<vector_RCP> & phi, // [set]
                                         vector<vector_RCP> & phi_stage, // [stage]
                                         vector<vector_RCP> & phi_prev) { // [step]
   // Goal is to update sol_stage[stage]
   // Assembler will need to gather sol for other physics sets and other step/stage solutions for current set

  Teuchos::TimeMonitor localtimer(*nonlinearsolvertimer);

  debugger->print(1, "******** Starting SolverManager::nonlinearSolver ...");

  int status = 0;
  int NLiter = 0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_first(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm(1);
  resnorm_first[0] = 10*NLtol;
  resnorm_scaled[0] = resnorm_first[0];
  resnorm[0] = resnorm_first[0];
  
  int maxiter = maxNLiter;
  if (is_adjoint) {
    maxiter = 1;//2;
  }
    
  bool proceed = true;
  ScalarT alpha = 1.0;
  
  vector_RCP current_res, current_res_over, current_du, current_du_over;
  if (store_vectors) {
    current_res = res[set];
    current_res_over = res_over[set];
    current_du = du[set];
    current_du_over = du_over[set];
  }
  else {
    current_res = linalg->getNewVector(set);
    current_res_over = linalg->getNewOverlappedVector(set);
    current_du = linalg->getNewVector(set);
    current_du_over = linalg->getNewOverlappedVector(set);
  }
  
  while (proceed) {
    
    multiscale_manager->reset();
    multiscale_manager->macro_nl_iter = NLiter;

    gNLiter = NLiter;
  
    bool build_jacobian = !linalg->getJacobianReuse(set);
    matrix_RCP J, J_over;

    J = linalg->getNewMatrix(set);
    if (build_jacobian) {
      J_over = linalg->getNewOverlappedMatrix(set);
      linalg->fillComplete(J_over);
      J_over->resumeFill();
      J_over->setAllToScalar(0.0);
    }
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    current_res_over->putScalar(0.0);

    store_adjPrev = false; //false;
    if ( is_adjoint && (NLiter == 1)) {
      store_adjPrev = true;
    }

    bool use_autotune = true;
    if (assembler->groupData[0]->multiscale) {
      use_autotune = false;
    }

    auto paramvec = params->getDiscretizedParamsOver();
    auto paramdot = params->getDiscretizedParamsDotOver();
    
    // This is where the residual is computed for the forward problem
    // Jacobian is computed only if the residual is large enough to merit a linear solve
    // Adjoint residual is computed below
    if (!is_adjoint) {
      if (!use_autotune) {
        assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, false, 0,
                                  current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                  params->num_active_params, paramvec, paramdot, is_final_time, deltat);
      }
      else {
        assembler->assembleRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev,
                               paramvec, paramdot, current_res_over, J_over, isTransient, current_time, deltat);
      }
      linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
    }
    
    // *********************** CHECK THE NORM OF THE RESIDUAL **************************
    
    {
      Teuchos::TimeMonitor localtimer(*normLAtimer);
      current_res->normInf(resnorm);
    }
    
    bool solve = true;
    if (NLiter == 0) {
      resnorm_first[0] = resnorm[0];
      resnorm_scaled[0] = 1.0;
    }
    else {
      resnorm_scaled[0] = resnorm[0]/resnorm_first[0];
    }
    
    //KokkosTools::print(current_res);
    // hard code these for adjoint solves since residual is computed below and only one iteration is needed
    if (is_adjoint) {
      resnorm[0] = 1.0;
      resnorm_scaled[0] = 1.0;
    }
    
    if (Comm->getRank() == 0 && verbosity > 1) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Iteration: " << NLiter << endl;
      cout << "***** Norm of nonlinear residual: " << resnorm[0] << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled[0] << endl;
      cout << "*********************************************************" << endl;
    }
    
    if (!is_adjoint && allowBacktracking && resnorm_scaled[0] > 1.1) {
      solve = false;
      alpha *= 0.5;
      Teuchos::TimeMonitor localtimer(*updateLAtimer);
      if (sol_stage.size() > 0) {
        sol_stage[stage]->update(-1.0*alpha, *(current_du_over), 1.0);
      }
      else {
        sol[set]->update(-1.0*alpha, *(current_du_over), 1.0);
      }
      if (Comm->getRank() == 0 && verbosity > 1) {
        cout << "***** Backtracking: new learning rate = " << alpha << endl;
      }
    }
    else {
      if (useRelativeTOL) {
        if (resnorm_scaled[0]<NLtol) {
          solve = false;
          proceed = false;
        }
        else if (resnorm[0]<1.0e-100) { // Not sure why this is hard coded
          solve = false;
          proceed = false;
        }
      }
      else if (useAbsoluteTOL && resnorm[0]<NLabstol) {
        solve = false;
        proceed = false;
      }
    }
    if (is_adjoint) { // Always perform one linear solve
      solve = true; // force a solve
      proceed = false; // but only one
    }
    
    // *********************** SOLVE THE LINEAR SYSTEM FOR THE UPDATE **************************
    
    if (solve) {
      
      if (build_jacobian) {
        if (use_autotune) { // If false, J was already computed when the residual was computed - just saves an extra assembly
          auto paramvec = params->getDiscretizedParamsOver();
          auto paramdot = params->getDiscretizedParamsDotOver();
          assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, false, 0,
                                    current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                    params->num_active_params, paramvec, paramdot, is_final_time, deltat);
        }
        linalg->fillComplete(J_over);
        J->resumeFill();
        linalg->exportMatrixFromOverlapped(set, J, J_over);
        linalg->fillComplete(J);
      }
            
      //KokkosTools::print(J);
      // This is where the adjoint residual is computed
      if (is_adjoint) {
        Teuchos::TimeMonitor localtimer(*transientadjointrhstimer);
        // First, the derivative of the objective w.r.t. the state
        ScalarT cdt = 0.0;
        if (isTransient) {
          cdt = deltat;
        }
        postproc->computeObjectiveGradState(set, sol[set], current_time+cdt, deltat, current_res);
        
        // We use a true adjoint residual, so we need to Jacobian^T times the current approximation
        
        auto mvprod = linalg->getNewVector(set);
        auto phi_owned = linalg->getNewVector(set);
        linalg->exportVectorFromOverlappedReplace(set, phi_owned, phi_stage[set]);
        J->apply(*phi_owned,*mvprod);
        current_res->update(-1.0, *mvprod, 1.0);
        
        
        // For transient problems, need to update adjoint residual with previous adjoint solutions (multi-step)
        // This is actually a little complicated
        // The jacobians need to be evaluated at the right time and using the right forward solution
        // The Jacobian-vector products we need on this time step should already be computed
        if (isTransient) {
          
          // use the stored Jacobian vector (Jv) products from previous time steps
          for (size_t istep=0; istep<previous_adjoints.size(); ++istep) {
            current_res->update(-1.0, *(previous_adjoints[istep][istep]), 1.0);
          }
          
          // Increment the Jv products
          size_t numSteps = sol_prev.size();
          for (size_t istep=0; istep<numSteps-1; ++istep) {
            for (size_t kstep=0; kstep<numSteps; ++kstep) {
              previous_adjoints[kstep][numSteps-istep] = previous_adjoints[kstep][numSteps-istep-1];
            }
          }
          
          // The next set of Jacobian vector products are calculated below once phi is updated
          
        }
        {
          Teuchos::TimeMonitor localtimer(*normLAtimer);
          current_res->normInf(resnorm);
        }
        
      }
      
      //******************************************************
      // Actual linear solve
      //******************************************************
      
      current_du->putScalar(0.0);
      current_du_over->putScalar(0.0);
      if (set < linalg->context.size() && !linalg->context[set].is_null()) {
        linalg->context[set]->jacobian_rebuilt_this_step = build_jacobian;
        linalg->context[set]->operator_dt = deltat / BDF_wts[set](0);
        linalg->context[set]->stage_alpha_u =
          (isTransient && butcher_b[set](stage) != 0.0)
            ? butcher_A[set](stage, stage) / butcher_b[set](stage) : 1.0;
      }
      linalg->linearSolver(set, J, current_res, current_du);
      
      // doesn't always write to file - only if requested
      if (is_adjoint) {
        linalg->writeToFile(J, current_res, current_du, "adjoint_jacobian.mm",
                            "adjoint_residual.mm","adjoint_solution.mm");
      }
      else {
        linalg->writeToFile(J, current_res, current_du);
      }
      linalg->importVectorToOverlapped(set, current_du_over, current_du);
      
      alpha = 1.0; // what is the point of alpha?
      if (is_adjoint) {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        if (phi_stage.size() > 0) {
          phi_stage[stage]->update(alpha, *(current_du_over), 1.0);
        }
        else {
          phi[set]->update(alpha, *(current_du_over), 1.0);
        }
        
        if (isTransient) {
          // Fill in prev_mass_adjoints[:][0] - need to create new vectors since these are RCPs
          bool build_prev_jacobian = !linalg->getPrevJacobianReuse(set);
          
          vector<matrix_RCP> Jprev = linalg->getNewPreviousMatrix(set, phi_prev.size());
          
          if (build_prev_jacobian) {
            
            for (size_t step=0; step<phi_prev.size(); ++step) {
              auto paramvec = params->getDiscretizedParamsOver();
              auto paramdot = params->getDiscretizedParamsDotOver();
              matrix_RCP currJ = Jprev[step];
              matrix_RCP currJ_over = linalg->getNewOverlappedMatrix(set);
              linalg->fillComplete(currJ_over);
              currJ_over->resumeFill();
              currJ->setAllToScalar(0.0);
              currJ_over->setAllToScalar(0.0);
              auto dummy_res_over = linalg->getNewOverlappedVector(set);
              assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, true, false, false, true, step,
                                        dummy_res_over, currJ_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                        params->num_active_params, paramvec, paramdot, is_final_time, deltat);
              linalg->fillComplete(currJ_over);
              currJ->resumeFill();
              linalg->exportMatrixFromOverlapped(set, currJ, currJ_over);
              linalg->fillComplete(currJ);
            }
          }
          
          for (size_t step=0; step<phi_prev.size(); ++step) {
          
            auto mvprod = linalg->getNewVector(set);
            auto phip_owned = linalg->getNewVector(set);
            
            if (step == 0) {
              if (phi_stage.size() > 0) {
                linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi_stage[stage]);
              }
              else {
                linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi[set]);
              }
            }
            else {
              linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi_prev[step-1]);
            }
            
            Jprev[step]->apply(*phip_owned,*mvprod);
            previous_adjoints[0][0]->update(1.0, *mvprod, 0.0);
            
          }
        }
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        if (sol_stage.size() > 0) {
          sol_stage[stage]->update(alpha, *(current_du_over), 1.0);
        }
        else {
          sol[set]->update(alpha, *(current_du_over), 1.0);
        }
        
      }
    }
    NLiter++; // increment number of nonlinear iterations
    
    if (is_adjoint) {
      //proceed = false;
    }
    else if (NLiter >= maxiter) {
      proceed = false;
    }
  } // while loop
  
  if (verbosity>1) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> normu(1);
    if (sol_stage.size() > 0) {
      sol_stage[stage]->norm2(normu);
    }
    else {
      sol[set]->norm2(normu);
    }
    if (Comm->getRank() == 0) {
      cout << "Norm of solution: " << normu[0] << "    (overlapped vector so results may differ on multiple procs)" << endl;
    }
  }
  if (Comm->getRank() == 0) {
    if (!is_adjoint) {
      if ( (NLiter>maxNLiter) && verbosity > 1) {
        status = 1;
        cout << endl << endl << "********************" << endl;
        cout << endl << "SOLVER FAILED TO CONVERGE CONVERGED in " << NLiter
        << " iterations with residual norm " << resnorm[0] << endl;
        cout << "********************" << endl;
      }
    }
  }
  
  debugger->print(1, "******** Finished SolverManager::nonlinearSolver");
  
  return status;
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::explicitSolver(const size_t & set, const size_t & stage,
                                        vector<vector_RCP> & sol, // [set]
                                        vector<vector_RCP> & sol_stage, // [stage]
                                        vector<vector_RCP> & sol_prev, // [step]
                                        vector<vector_RCP> & phi, // [set]
                                        vector<vector_RCP> & phi_stage, // [stage]
                                        vector<vector_RCP> & phi_prev) { // [step]
  
  
  // Goal is just to update sol_stage[stage]
  // Other solutions are just used in assembler for gather operations

  Teuchos::TimeMonitor localtimer(*explicitsolvertimer);
  
  debugger->print(1, "******** Starting SolverManager::explicitSolver ...");
  
  int status = 0;
  assembler->updatePhysicsSet(set);
  
  if (usestrongDBCs) {
    this->setDirichlet(set,sol_stage[stage]);
  }
  
  vector_RCP current_res, current_res_over;
  if (store_vectors) {
    current_res = res[set];
    current_res_over = res_over[set];
  }
  else {
    current_res = linalg->getNewVector(set);
    if (linalg->getHaveOverlapped()) {
      current_res_over = linalg->getNewOverlappedVector(set);
    }
    else {
      current_res_over = current_res;
    }
  }

  // *********************** COMPUTE THE RESIDUAL **************************
    
  current_res_over->putScalar(0.0);
  matrix_RCP J_over;
  
  auto paramvec = params->getDiscretizedParamsOver();
  auto paramdot = params->getDiscretizedParamsDotOver();
  assembler->assembleRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev,
                         paramvec, paramdot, current_res_over, J_over, isTransient, current_time, deltat);
  
  if (linalg->getHaveOverlapped()) {
    linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
  }

  // The mass solve keeps fixed DOFs only if res is zero there. The scatter skips rows a rank
  // flags as fixed, but a rank that ghosts a fixed DOF without owning its boundary side does not
  // flag it, and its contribution is summed in by the export. Zero the owned fixed rows here.
  {
    typedef typename Node::execution_space LA_exec;
    auto fixed = explicitFixedDOF[set];
    auto res_view = current_res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    parallel_for("explicit solver mask res",
                 RangePolicy<LA_exec>(0,res_view.extent(0)),
                 MRHYDE_LAMBDA (const int k ) {
      if (fixed(k)) {
        res_view(k,0) = 0.0;
      }
    });
  }

  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  current_res->norm2(rnorm);
  //KokkosTools::print(current_res);
  
  // *********************** SOLVE THE LINEAR SYSTEM **************************
  
  if (rnorm[0]>1.0e-100) {
    // Given m = diag(M^-1)
    // Given timewt = b(stage)*deltat
    // Compute du = timewt*m*res
    // Compute u += du
    
    ScalarT wt = deltat*butcher_b[set](stage);
    
    
    if (!assembler->lump_mass) {
      vector_RCP current_du, current_du_over;
      if (store_vectors) {
        current_du = du[set];
        current_du_over = du_over[set];
      }
      else {
        current_du = linalg->getNewVector(set);
        if (linalg->getHaveOverlapped()) {
          current_du_over = linalg->getNewOverlappedVector(set);
        }
        else {
          current_du_over = current_du;
        }
      }

      current_du_over->putScalar(0.0);
      if (linalg->getHaveOverlapped()) {
        current_du->putScalar(0.0);
      }

      current_res->scale(wt);
      if (assembler->matrix_free) {
        this->matrixFreePCG(set, current_res, current_du, diagMass[set],
                            settings->sublist("Solver").get("linear TOL",1.0e-2),
                            settings->sublist("Solver").get("max linear iters",100));
      }
      else {
        if (use_custom_PCG) {
          this->PCG(set, explicitMass[set], current_res, current_du, diagMass[set],
                    settings->sublist("Solver").get("linear TOL",1.0e-2),
                    settings->sublist("Solver").get("max linear iters",100));
        }
        else {
          linalg->linearSolverL2(set, explicitMass[set], current_res, current_du);
        }
      }
      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, current_du_over, current_du);
      }
      
      sol_stage[stage]->update(1.0, *(current_du_over), 1.0);
      
    }
    else {
      typedef typename Node::execution_space LA_exec;
      
      // can probably avoid du in this case
      // sol += sol + wt*res/dm
      vector_RCP current_sol;
      if (linalg->getHaveOverlapped()) {
        current_sol = linalg->getNewVector(set);
        linalg->exportVectorFromOverlappedReplace(set, current_sol, sol_stage[stage]);
      }
      else {
        current_sol = sol_stage[stage];
      }

      //auto du_view = current_du->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto sol_view = current_sol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto res_view = current_res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto dm_view = diagMass[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      parallel_for("explicit solver apply invdiag",
                   RangePolicy<LA_exec>(0,sol_view.extent(0)),
                   MRHYDE_LAMBDA (const int k ) {
        sol_view(k,0) += wt*res_view(k,0)/dm_view(k,0);
      });

      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, sol_stage[stage], current_sol);
      }
    }
    
  }
  
  if (verbosity>=10) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> unorm(1);
    sol_stage[stage]->norm2(unorm);
    if (Comm->getRank() == 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Explicit integrator: L2 norm of (overlapped/ghosted) solution: " << unorm[0] << endl;
      cout << "*********************************************************" << endl;
    }
  }
  
  debugger->print(1, "******** Finished SolverManager::explicitSolver");

  return status;
}

// ========================================================================================
// x = deltat*b(0)*M_set^{-1} rhs, using the same mass operator as explicitSolver
// ========================================================================================

template<class Node>
void SolverManager<Node>::applyExplicitMassInverse(const size_t & set, vector_RCP & rhs, vector_RCP & x) {

  // The (matrix-free) mass operators use the current physics set of the assembler
  assembler->updatePhysicsSet(set);

  ScalarT wt = deltat*butcher_b[set](0);
  x->putScalar(0.0);

  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  rhs->norm2(rnorm);
  if (rnorm[0] <= 1.0e-100) {
    return;
  }

  if (!assembler->lump_mass) {
    rhs->scale(wt);
    if (assembler->matrix_free) {
      this->matrixFreePCG(set, rhs, x, diagMass[set],
                          settings->sublist("Solver").get("linear TOL",1.0e-2),
                          settings->sublist("Solver").get("max linear iters",100));
    }
    else {
      if (use_custom_PCG) {
        this->PCG(set, explicitMass[set], rhs, x, diagMass[set],
                  settings->sublist("Solver").get("linear TOL",1.0e-2),
                  settings->sublist("Solver").get("max linear iters",100));
      }
      else {
        linalg->linearSolverL2(set, explicitMass[set], rhs, x);
      }
    }
  }
  else {
    typedef typename Node::execution_space LA_exec;
    auto x_view = x->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto rhs_view = rhs->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto dm_view = diagMass[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    parallel_for("explicit adjoint apply invdiag",
                 RangePolicy<LA_exec>(0,x_view.extent(0)),
                 MRHYDE_LAMBDA (const int k ) {
      x_view(k,0) = wt*rhs_view(k,0)/dm_view(k,0);
    });
  }
}

// ========================================================================================
// Linear part of the explicit residual: res = r(x,y) - r(0,0) with r = -R (what assembleRes
// returns), zero discretized parameters and zero time derivative for the current set
// ========================================================================================

template<class Node>
void SolverManager<Node>::explicitLinearResidual(const size_t & set, vector_RCP & x_over, vector_RCP & y_over,
                                                 vector_RCP & res, const bool & linear_part,
                                                 vector_RCP p_over) {

  assembler->updatePhysicsSet(set);

  vector_RCP pzero;
  if (params->num_discretized_params > 0) {
    pzero = linalg->getNewOverlappedParamVector();
    pzero->putScalar(0.0);
  }
  else {
    pzero = params->getDiscretizedParamsOver();
  }

  auto evalRes = [&](vector_RCP & u0, vector_RCP & u1, vector_RCP & pvec, vector_RCP & out) {
    vector<vector_RCP> lsol = {u0, u1};
    // stage = prev = current state, so the time derivative of this set vanishes (as in explicitSolver)
    vector<vector_RCP> lstage = {lsol[set]};
    vector<vector_RCP> lprev = {lsol[set]};
    vector<vector_RCP> zero_vec;
    vector_RCP res_over = linalg->getNewOverlappedVector(set);
    res_over->putScalar(0.0);
    matrix_RCP J_over;
    assembler->assembleRes(set, 0, lsol, lstage, lprev, zero_vec, zero_vec, zero_vec,
                           pvec, pzero, res_over, J_over, isTransient, current_time, deltat);
    if (linalg->getHaveOverlapped()) {
      linalg->exportVectorFromOverlapped(set, out, res_over);
    }
    else {
      out->assign(*res_over);
    }
  };

  vector_RCP zero0 = linalg->getNewOverlappedVector(0);
  vector_RCP zero1 = linalg->getNewOverlappedVector(1);
  zero0->putScalar(0.0);
  zero1->putScalar(0.0);

  if (Teuchos::is_null(p_over)) {
    p_over = pzero;
  }
  if (!linear_part) {
    evalRes(x_over, y_over, p_over, res);
    return;
  }

  // r is affine: r(0,0) holds the physical source. Adjoint vectors can be many orders of magnitude
  // below the source, so r(x,y) - r(0,0) would cancel catastrophically. Evaluate at s*(x,y,p) with
  // max norm 1e100 instead, so the linear part dominates, and divide by s.
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> nrm(1);
  ScalarT maxnorm = 0.0;
  x_over->normInf(nrm);
  maxnorm = std::max(maxnorm, (ScalarT)nrm[0]);
  y_over->normInf(nrm);
  maxnorm = std::max(maxnorm, (ScalarT)nrm[0]);
  p_over->normInf(nrm);
  maxnorm = std::max(maxnorm, (ScalarT)nrm[0]);
  if (maxnorm < std::numeric_limits<ScalarT>::min()) {
    res->putScalar(0.0);
    return;
  }
  // Scale in two steps (normalize, then 1e100) and undo it in two steps: a single factor
  // 1e100/maxnorm overflows for maxnorm < ~1e-208, and maxnorm*1e-100 underflows likewise
  ScalarT inv_norm = 1.0/maxnorm;
  vector_RCP xs = linalg->getNewOverlappedVector(0), ys = linalg->getNewOverlappedVector(1);
  xs->update(inv_norm, *x_over, 0.0);
  xs->scale(1.0e100);
  ys->update(inv_norm, *y_over, 0.0);
  ys->scale(1.0e100);
  vector_RCP ps = pzero;
  if (p_over != pzero) {
    ps = linalg->getNewOverlappedParamVector();
    ps->update(inv_norm, *p_over, 0.0);
    ps->scale(1.0e100);
  }
  vector_RCP res0 = linalg->getNewVector(set);
  evalRes(xs, ys, ps, res);
  evalRes(zero0, zero1, pzero, res0);
  res->update(-1.0, *res0, 1.0);
  res->scale(1.0e-100);
  res->scale(maxnorm);
}

// ========================================================================================
// AD-free discrete adjoint of the fully explicit E/B split
//
// Forward (E = set 0, B = set 1, one FWE stage each, r = -R from assembleRes):
//   E^{n+1} = E^n + dt M_E^{-1} r_E(E^n, B^n, p_n, t_n)
//   B^{n+1} = B^n + dt M_B^{-1} r_B(E^{n+1})
// with (3D maxwell) R_E = (1/eps)(S E + F p + g) - 1/(mu eps) K B and R_B = C E, C = K^T.
// Objective (integrated control, t=0 skipped):
//   J = sum_{n=1}^N dt w int_obs 0.5 eps |E^n|^2 + 0.5/mu |B^n|^2.
// Backward sweep, n = N,...,1, with Ebar^{N+1} = Bbar^{N+1} = mu_E^{N+1} = 0:
//   Bbar^n  = dJ/dB^n + Bbar^{n+1} - 1/(mu eps) r_B(mu_E^{n+1})          [= ... + 1/(mu eps) K^T mu_E]
//   mu_B^n  = dt M_B^{-1} Bbar^n
//   Ebar^n  = dJ/dE^n + Ebar^{n+1} + r_E^lin(mu_E^{n+1}, -mu eps mu_B^n)  [= ... - (1/eps) S mu_E - C^T mu_B]
//   mu_E^n  = dt M_E^{-1} Ebar^n
//   dJ/dp_{n-1} = -(1/eps) F^T mu_E^n
// r^lin is the residual with zero control and zero source (explicitLinearResidual, which evaluates it at
// scaled arguments so the source does not swamp the tiny adjoint vectors).
// Requires constant eps and mu, no strong Dirichlet or Neumann (ABC) conditions, and the
// EM-energy objective.
// ========================================================================================

template<class Node>
void SolverManager<Node>::explicitSplitAdjoint(MrHyDE_OptVector & gradient) {

  debugger->print(1, "******** Starting SolverManager::explicitSplitAdjoint ...");

  // ---------------------------- checks ----------------------------

  TEUCHOS_TEST_FOR_EXCEPTION(dimension != 3, std::runtime_error,
                             "Error: the explicit adjoint requires a 3D mesh (the 2D maxwell residual is not scaled by 1/eps).");
  TEUCHOS_TEST_FOR_EXCEPTION(setnames.size() != 2, std::runtime_error,
                             "Error: the explicit adjoint requires exactly two physics sets (E, B).");
  {
    auto & vlist = assembler->wkset[0]->set_varlist;
    bool varsok = (vlist[0].size() == 1 && vlist[0][0] == "E" && vlist[1].size() == 1 && vlist[1][0] == "B");
    TEUCHOS_TEST_FOR_EXCEPTION(!varsok, std::runtime_error,
                               "Error: the explicit adjoint requires set 0 = {E} and set 1 = {B}.");
  }
  for (size_t set=0; set<setnames.size(); ++set) {
    TEUCHOS_TEST_FOR_EXCEPTION(numstages[set] != 1 || maxnumstages[set] != 1, std::runtime_error,
                               "Error: the explicit adjoint requires a single-stage scheme (FWE) in each set.");
    TEUCHOS_TEST_FOR_EXCEPTION(BDForder[set] != 1, std::runtime_error,
                               "Error: the explicit adjoint requires 'transient BDF order: 1'.");
    TEUCHOS_TEST_FOR_EXCEPTION(std::abs(butcher_A[set](0,0)) > 1.0e-14, std::runtime_error,
                               "Error: the explicit adjoint requires an explicit stage (Butcher A(0,0) = 0).");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(usestrongDBCs, std::runtime_error,
                             "Error: the explicit adjoint does not support strong Dirichlet conditions.");
  TEUCHOS_TEST_FOR_EXCEPTION(postproc->is_incremental_adjoint, std::runtime_error,
                             "Error: incremental (second-order) adjoints are not implemented for the explicit split.");
  for (size_t block=0; block<assembler->wkset.size(); ++block) {
    auto bcs = assembler->wkset[block]->set_var_bcs[1];
    for (size_type i=0; i<bcs.extent(0); ++i) {
      for (size_type j=0; j<bcs.extent(1); ++j) {
        TEUCHOS_TEST_FOR_EXCEPTION(bcs(i,j) == "Neumann", std::runtime_error,
                                   "Error: the explicit adjoint does not support Neumann (ABC) conditions on B.");
      }
    }
  }

  // eps and mu must be the same numeric constant in every block (C = K^T up to 1/(mu eps))
  auto blockConstant = [&](const string & name) {
    ScalarT val = 0.0;
    for (size_t block=0; block<blocknames.size(); ++block) {
      Teuchos::ParameterList & funcs = settings->sublist("Functions");
      Teuchos::ParameterList fs = funcs.isSublist(blocknames[block]) ? funcs.sublist(blocknames[block]) : funcs;
      string sval = fs.get<string>(name,"1.0");
      ScalarT bval = 0.0;
      size_t pos = 0;
      try {
        bval = std::stod(sval, &pos);
      }
      catch (...) {
        pos = 0;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(pos == 0 || sval.find_first_not_of(" ", pos) != string::npos, std::runtime_error,
                                 "Error: the explicit adjoint requires '" + name + "' to be a numeric constant, got '" + sval + "' on " + blocknames[block]);
      if (block == 0) {
        val = bval;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(std::abs(bval-val) > 1.0e-12*std::abs(val), std::runtime_error,
                                 "Error: the explicit adjoint requires '" + name + "' to be the same in every block.");
    }
    return val;
  };
  ScalarT eps = blockConstant("permittivity");
  ScalarT mu = blockConstant("permeability");

  auto & soln_source = postproc->soln;
  int store_index = 0;
  size_t numFwdSteps = soln_source[0]->getTotalTimes(store_index)-1;
  TEUCHOS_TEST_FOR_EXCEPTION(soln_source[1]->getTotalTimes(store_index)-1 != numFwdSteps, std::runtime_error,
                             "Error: the explicit adjoint requires the forward solution of both sets at every time step.");
  for (size_t n=1; n<=numFwdSteps; ++n) {
    ScalarT dtn = soln_source[0]->getSpecificTime(store_index, n) - soln_source[0]->getSpecificTime(store_index, n-1);
    TEUCHOS_TEST_FOR_EXCEPTION(std::abs(dtn-deltat) > 1.0e-8*deltat, std::runtime_error,
                               "Error: the explicit adjoint requires a fixed time step (no time step cuts).");
  }

  // ---------------------------- helpers ----------------------------

  bool have_over = linalg->getHaveOverlapped();
  auto toOverlapped = [&](const size_t & set, vector_RCP & over, vector_RCP & owned) {
    if (have_over) {
      linalg->importVectorToOverlapped(set, over, owned);
    }
    else {
      over->assign(*owned);
    }
  };
  auto extractState = [&](const size_t & set, vector_RCP & vec, const size_t & index) {
    bool fnd = soln_source[set]->extract(vec, index);
    TEUCHOS_TEST_FOR_EXCEPTION(!fnd, std::runtime_error, "Error: MrHyDE was not able to find the forward solution for the explicit adjoint");
  };

  // Forward states (overlapped), adjoint vectors (owned) and mass-scaled adjoints (overlapped)
  vector_RCP E_n = linalg->getNewOverlappedVector(0), B_n = linalg->getNewOverlappedVector(1);
  vector_RCP Ebar = linalg->getNewVector(0), Ebar_next = linalg->getNewVector(0);
  vector_RCP Bbar = linalg->getNewVector(1), Bbar_next = linalg->getNewVector(1);
  vector_RCP muE = linalg->getNewVector(0), muB = linalg->getNewVector(1);
  vector_RCP muE_over = linalg->getNewOverlappedVector(0), muE_next_over = linalg->getNewOverlappedVector(0);
  vector_RCP muB_over = linalg->getNewOverlappedVector(1), zero_B_over = linalg->getNewOverlappedVector(1);
  vector_RCP rhsE = linalg->getNewVector(0), rhsB = linalg->getNewVector(1);
  vector_RCP resE = linalg->getNewVector(0), resB = linalg->getNewVector(1);
  vector_RCP sens;
  if (params->num_discretized_params > 0) {
    sens = linalg->getNewParamVector();
  }
  Ebar_next->putScalar(0.0);
  Bbar_next->putScalar(0.0);
  muE_next_over->putScalar(0.0);
  zero_B_over->putScalar(0.0);

  // Optional checks ('explicit adjoint operator check: true' in Solver) of what the sweep relies on.
  // With random vectors x, y, d (and a, b in either set):
  //   C = K^T:      -<r_B(x), y> = mu eps <x, r_E^lin(0, y)>
  //   S symmetric:  <r_E^lin(x1, 0), x2> = <x1, r_E^lin(x2, 0)>
  //   F^T:          <-(1/eps) F^T x, d> = <x, r_E^lin(0, 0; d)>
  //   M symmetric:  <dt M^{-1} a, b> = <a, dt M^{-1} b>
  //   CFL:          dt^2 lambda_max(M_E^{-1} K M_B^{-1} C)/(mu eps) <= 4 (symplectic Euler stability)
  // With the stored forward states at n = N/2:
  //   forward step: E^{n+1} - E^n = dt M_E^{-1} r_E(E^n, B^n; p_n),  B^{n+1} - B^n = dt M_B^{-1} r_B(E^{n+1})
  //   objective:    <dJ/dE^n, E^n> + <dJ/dB^n, B^n> = 2 dt J_n  (J_n is quadratic)
  // and, after the sweep, a tangent test of the whole adjoint (see below).
  if (settings->sublist("Solver").get<bool>("explicit adjoint operator check",false)) {
    auto randOver = [&](const size_t & set, vector_RCP & owned, vector_RCP & over) {
      owned->randomize();
      toOverlapped(set, over, owned);
    };
    auto dotp = [](vector_RCP & a, vector_RCP & b) {
      Teuchos::Array<ScalarT> d(1);
      a->dot(*b, d);
      return d[0];
    };
    vector_RCP x1 = linalg->getNewVector(0), x2 = linalg->getNewVector(0), y1 = linalg->getNewVector(1), y2 = linalg->getNewVector(1);
    vector_RCP x1o = linalg->getNewOverlappedVector(0), x2o = linalg->getNewOverlappedVector(0);
    vector_RCP y1o = linalg->getNewOverlappedVector(1), zero_E_over = linalg->getNewOverlappedVector(0);
    zero_E_over->putScalar(0.0);
    randOver(0, x1, x1o);
    randOver(0, x2, x2o);
    randOver(1, y1, y1o);
    y2->randomize();
    current_time = soln_source[0]->getSpecificTime(store_index, 0);
    assembler->updateStage(0, current_time, deltat);

    this->explicitLinearResidual(1, x1o, zero_B_over, resB);
    ScalarT lhsC = -dotp(resB, y1);
    this->explicitLinearResidual(0, zero_E_over, y1o, resE);
    ScalarT rhsC = mu*eps*dotp(x1, resE);

    this->explicitLinearResidual(0, x1o, zero_B_over, resE);
    ScalarT lhsS = dotp(resE, x2);
    this->explicitLinearResidual(0, x2o, zero_B_over, resE);
    ScalarT rhsS = dotp(x1, resE);

    ScalarT lhsF = 0.0, rhsF = 0.0;
    if (params->num_discretized_params > 0) {
      vector_RCP d = linalg->getNewParamVector(), d_over = linalg->getNewOverlappedParamVector();
      d->randomize();
      d_over->putScalar(0.0);
      d_over->doImport(*d, *(linalg->param_importer), Tpetra::INSERT);
      vector_RCP s = linalg->getNewParamVector();
      postproc->computeControlTranspose(x1o, -1.0/eps, s);
      lhsF = dotp(s, d);
      this->explicitLinearResidual(0, zero_E_over, zero_B_over, resE, true, d_over);
      rhsF = dotp(x1, resE);
    }
    
    // CFL: power iteration on x <- -A_E r_E^lin(0, A_B r_B(x)) = dt^2/(mu eps) M_E^{-1} K M_B^{-1} C x
    ScalarT lam = 0.0;
    {
      vector_RCP x = linalg->getNewVector(0), xo = linalg->getNewOverlappedVector(0);
      vector_RCP y = linalg->getNewVector(1), yo = linalg->getNewOverlappedVector(1), z = linalg->getNewVector(0);
      randOver(0, x, xo);
      for (int it=0; it<300; ++it) {
        Teuchos::Array<ScalarT> nx(1), nz(1);
        x->norm2(nx);
        this->explicitLinearResidual(1, xo, zero_B_over, resB);
        this->applyExplicitMassInverse(1, resB, y);
        toOverlapped(1, yo, y);
        this->explicitLinearResidual(0, zero_E_over, yo, resE);
        this->applyExplicitMassInverse(0, resE, z);
        z->scale(-1.0);
        z->norm2(nz);
        lam = nz[0]/nx[0];
        x->assign(*z);
        x->scale(1.0/nz[0]);
        toOverlapped(0, xo, x);
      }
    }

    vector<ScalarT> lhsM(2), rhsM(2);
    for (size_t set=0; set<2; ++set) {
      vector_RCP a = (set == 0) ? x1 : y1, b = (set == 0) ? x2 : y2;
      vector_RCP r = linalg->getNewVector(set), xa = linalg->getNewVector(set), xb = linalg->getNewVector(set);
      r->assign(*a);
      this->applyExplicitMassInverse(set, r, xa);
      r->assign(*b);
      this->applyExplicitMassInverse(set, r, xb);
      lhsM[set] = dotp(xa, b);
      rhsM[set] = dotp(a, xb);
    }
    // Forward step and objective at n = N/2
    ScalarT stepErr[2] = {0.0, 0.0}, stepNorm[2] = {0.0, 0.0}, objCheck[2] = {0.0, 0.0};
    {
      size_t n = numFwdSteps/2;
      vector_RCP En = linalg->getNewOverlappedVector(0), Bn = linalg->getNewOverlappedVector(1);
      vector_RCP Enp = linalg->getNewOverlappedVector(0), Bnp = linalg->getNewOverlappedVector(1);
      extractState(0, En, n);
      extractState(1, Bn, n);
      extractState(0, Enp, n+1);
      extractState(1, Bnp, n+1);
      current_time = soln_source[0]->getSpecificTime(store_index, n);
      assembler->updateStage(0, current_time, deltat);
      {
        vector_RCP gE = linalg->getNewVector(0), gB = linalg->getNewVector(1);
        vector_RCP Eo = linalg->getNewVector(0), Bo = linalg->getNewVector(1);
        gE->putScalar(0.0);
        gB->putScalar(0.0);
        postproc->addEnergyObjectiveGradState(0, En, deltat*eps, gE);
        postproc->addEnergyObjectiveGradState(1, Bn, deltat/mu, gB);
        if (have_over) {
          linalg->exportVectorFromOverlappedReplace(0, Eo, En);
          linalg->exportVectorFromOverlappedReplace(1, Bo, Bn);
        }
        else {
          Eo->assign(*En);
          Bo->assign(*Bn);
        }
        objCheck[0] = dotp(gE, Eo) + dotp(gB, Bo);
        ScalarT lval = 0.0, gval = 0.0;
        for (size_t r=0; r<postproc->objectives.size(); ++r) {
          if (postproc->objectives[r].objective_values.size() > n) {
            lval += postproc->objectives[r].objective_values[n];
          }
        }
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lval, &gval);
        objCheck[1] = 2.0*deltat*gval;
      }
      for (size_t set=0; set<2; ++set) {
        vector_RCP r = linalg->getNewVector(set), du = linalg->getNewVector(set);
        vector_RCP diff = linalg->getNewVector(set), tmp = linalg->getNewVector(set);
        if (set == 0) {
          params->updateDynamicParams(n);
          vector_RCP p_n;
          if (params->num_discretized_params > 0) {
            p_n = params->getDiscretizedParamsOver();
          }
          this->explicitLinearResidual(0, En, Bn, r, false, p_n);
        }
        else {
          this->explicitLinearResidual(1, Enp, zero_B_over, r, false);
        }
        this->applyExplicitMassInverse(set, r, du);
        vector_RCP a = (set == 0) ? Enp : Bnp, b = (set == 0) ? En : Bn;
        if (have_over) {
          linalg->exportVectorFromOverlappedReplace(set, diff, a);
          linalg->exportVectorFromOverlappedReplace(set, tmp, b);
        }
        else {
          diff->assign(*a);
          tmp->assign(*b);
        }
        diff->update(-1.0, *tmp, 1.0);
        Teuchos::Array<ScalarT> nrm(1);
        diff->norm2(nrm);
        stepNorm[set] = nrm[0];
        diff->update(-1.0, *du, 1.0);
        diff->norm2(nrm);
        stepErr[set] = nrm[0];
      }
    }
    if (Comm->getRank() == 0) {
      auto rel = [](ScalarT a, ScalarT b) { return std::abs(a-b)/std::max(std::abs(a)+std::abs(b), 1.0e-300); };
      cout.precision(10);
      cout << endl << "[EXPLICIT-ADJOINT-CHECK] forward E step : |dE - dt M^-1 r_E| = " << stepErr[0] << "  |dE| = " << stepNorm[0] << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] forward B step : |dB - dt M^-1 r_B| = " << stepErr[1] << "  |dB| = " << stepNorm[1] << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] objective grad : " << objCheck[0] << "  " << objCheck[1] << "  rel diff " << rel(objCheck[0],objCheck[1]) << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] CFL number     : " << lam << "  (stable iff <= 4)" << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] C = K^T      : " << lhsC << "  " << rhsC << "  rel diff " << rel(lhsC,rhsC) << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] S symmetric  : " << lhsS << "  " << rhsS << "  rel diff " << rel(lhsS,rhsS) << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] F^T          : " << lhsF << "  " << rhsF << "  rel diff " << rel(lhsF,rhsF) << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] M_E^-1 symm. : " << lhsM[0] << "  " << rhsM[0] << "  rel diff " << rel(lhsM[0],rhsM[0]) << endl;
      cout << "[EXPLICIT-ADJOINT-CHECK] M_B^-1 symm. : " << lhsM[1] << "  " << rhsM[1] << "  rel diff " << rel(lhsM[1],rhsM[1]) << endl << endl;
    }
  }

  // ---------------------------- backward sweep ----------------------------

  for (size_t timeiter=0; timeiter<numFwdSteps; ++timeiter) {

    size_t n = numFwdSteps-timeiter;

    if (Comm->getRank() == 0 && verbosity > 0) {
      cout << endl << endl << "*******************************************************" << endl;
      cout << endl << "**** Beginning Explicit Split Adjoint Time Step " << timeiter << " (forward state " << n << ")" << endl;
      cout << "*******************************************************" << endl << endl << endl;
    }

    extractState(0, E_n, n);
    extractState(1, B_n, n);

    // The linear residual parts do not depend on time; t_n is the time of the forward step n -> n+1
    current_time = soln_source[0]->getSpecificTime(store_index, n);
    postproc->setTimeIndex(n);
    assembler->updateStage(0, current_time, deltat);

    // Bbar^n = dJ/dB^n + Bbar^{n+1} - 1/(mu eps) r_B(mu_E^{n+1})
    Bbar->putScalar(0.0);
    postproc->addEnergyObjectiveGradState(1, B_n, deltat/mu, Bbar);
    Bbar->update(1.0, *Bbar_next, 1.0);
    this->explicitLinearResidual(1, muE_next_over, zero_B_over, resB);
    Bbar->update(-1.0/(mu*eps), *resB, 1.0);

    // mu_B^n = dt M_B^{-1} Bbar^n
    rhsB->assign(*Bbar);
    this->applyExplicitMassInverse(1, rhsB, muB);
    toOverlapped(1, muB_over, muB);

    // Ebar^n = dJ/dE^n + Ebar^{n+1} + r_E^lin(mu_E^{n+1}, -mu eps mu_B^n)
    Ebar->putScalar(0.0);
    postproc->addEnergyObjectiveGradState(0, E_n, deltat*eps, Ebar);
    Ebar->update(1.0, *Ebar_next, 1.0);
    muB_over->scale(-mu*eps);
    this->explicitLinearResidual(0, muE_next_over, muB_over, resE);
    Ebar->update(1.0, *resE, 1.0);

    // mu_E^n = dt M_E^{-1} Ebar^n
    rhsE->assign(*Ebar);
    this->applyExplicitMassInverse(0, rhsE, muE);
    toOverlapped(0, muE_over, muE);

    // dJ/dp_{n-1} = -(1/eps) F^T mu_E^n
    if (params->num_discretized_params > 0) {
      postproc->computeControlTranspose(muE_over, -1.0/eps, sens);
      auto disc_grad = gradient.getField();
      vector_RCP curr_grad = gradient.haveDynamicField() ? disc_grad[n-1]->getVector() : disc_grad[0]->getVector();
      curr_grad->update(1.0, *sens, 1.0);
    }

    std::swap(Ebar, Ebar_next);
    std::swap(Bbar, Bbar_next);
    std::swap(muE_over, muE_next_over);

    is_final_time = false;
  }

  // Optional tangent test of the whole sweep: for random dp_k, propagate the linearized forward
  //   dE^{n+1} = dE^n + dt M_E^{-1} (r_E^lin(dE^n, dB^n) + r_E^lin(0, 0; dp_n)),  dB^{n+1} = dB^n + dt M_B^{-1} r_B(dE^{n+1})
  // and compare dJ = sum_n <dJ/dE^n, dE^n> + <dJ/dB^n, dB^n> with sum_k <g_k, dp_k>
  if (settings->sublist("Solver").get<bool>("explicit adjoint operator check",false) && params->num_discretized_params > 0) {
    auto dotp = [](vector_RCP & a, vector_RCP & b) {
      Teuchos::Array<ScalarT> d(1);
      a->dot(*b, d);
      return d[0];
    };
    auto disc_grad = gradient.getField();
    vector_RCP dE = linalg->getNewVector(0), dB = linalg->getNewVector(1);
    vector_RCP dE_over = linalg->getNewOverlappedVector(0), dB_over = linalg->getNewOverlappedVector(1);
    vector_RCP zero_E_over = linalg->getNewOverlappedVector(0);
    vector_RCP du = linalg->getNewVector(0), dub = linalg->getNewVector(1), tmpE = linalg->getNewVector(0);
    vector_RCP gE = linalg->getNewVector(0), gB = linalg->getNewVector(1);
    vector_RCP dp = linalg->getNewParamVector(), dp_over = linalg->getNewOverlappedParamVector();
    dE->putScalar(0.0);
    dB->putScalar(0.0);
    dE_over->putScalar(0.0);
    dB_over->putScalar(0.0);
    zero_E_over->putScalar(0.0);
    ScalarT dJ = 0.0, gdp = 0.0, dJabs = 0.0, gdpabs = 0.0;
    for (size_t n=0; n<numFwdSteps; ++n) {
      dp->randomize();
      dp_over->putScalar(0.0);
      dp_over->doImport(*dp, *(linalg->param_importer), Tpetra::INSERT);
      vector_RCP gk = gradient.haveDynamicField() ? disc_grad[n]->getVector() : disc_grad[0]->getVector();
      ScalarT gterm = dotp(gk, dp);
      gdp += gterm;
      gdpabs += std::abs(gterm);

      current_time = soln_source[0]->getSpecificTime(store_index, n);
      assembler->updateStage(0, current_time, deltat);
      this->explicitLinearResidual(0, dE_over, dB_over, resE);
      this->explicitLinearResidual(0, zero_E_over, zero_B_over, tmpE, true, dp_over);
      resE->update(1.0, *tmpE, 1.0);
      this->applyExplicitMassInverse(0, resE, du);
      dE->update(1.0, *du, 1.0);
      toOverlapped(0, dE_over, dE);
      this->explicitLinearResidual(1, dE_over, zero_B_over, resB);
      this->applyExplicitMassInverse(1, resB, dub);
      dB->update(1.0, *dub, 1.0);
      toOverlapped(1, dB_over, dB);

      extractState(0, E_n, n+1);
      extractState(1, B_n, n+1);
      gE->putScalar(0.0);
      gB->putScalar(0.0);
      postproc->addEnergyObjectiveGradState(0, E_n, deltat*eps, gE);
      postproc->addEnergyObjectiveGradState(1, B_n, deltat/mu, gB);
      ScalarT jterm = dotp(gE, dE) + dotp(gB, dB);
      dJ += jterm;
      dJabs += std::abs(jterm);
    }
    if (Comm->getRank() == 0) {
      cout.precision(10);
      cout << endl << "[EXPLICIT-ADJOINT-CHECK] tangent test: dJ = " << dJ << "  sum_k <g_k,dp_k> = " << gdp
           << "  rel diff " << std::abs(dJ-gdp)/std::max(std::abs(dJ)+std::abs(gdp),1.0e-300)
           << "  (sum|terms|: " << dJabs << ", " << gdpabs << ")" << endl << endl;
    }
  }

  debugger->print(1, "******** Finished SolverManager::explicitSplitAdjoint");
}

// ========================================================================================
// Specialized PCG
// ========================================================================================

template<class Node>
void SolverManager<Node>::PCG(const size_t & set, matrix_RCP & J, vector_RCP & b, vector_RCP & x,
                              vector_RCP & M, const ScalarT & tol, const int & maxiter) {
  
  Teuchos::TimeMonitor localtimer(*PCGtimer);
  
  typedef typename Node::execution_space LA_exec;
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
  
  ScalarT rho = 1.0, rho1 = 0.0, alpha = 0.0, beta = 1.0, pq = 0.0;
  ScalarT one = 1.0, zero = 0.0;
  
  vector_RCP p, q, r, z;
  if (store_vectors) {
    p = p_pcg[set];
    q = q_pcg[set];
    r = r_pcg[set];
    z = z_pcg[set];
  }
  else {
    p = linalg->getNewVector(set);
    q = linalg->getNewVector(set);
    r = linalg->getNewVector(set);
    z = linalg->getNewVector(set);
  }
  
  p->putScalar(zero);
  q->putScalar(zero);
  r->putScalar(zero);
  z->putScalar(zero);
  
  int iter=0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  {
    Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
    J->apply(*x,*q);
  }
  
  r->assign(*b);
  r->update(-one,*q,one);
  
  r->norm2(rnorm);
  ScalarT r0 = rnorm[0];
  
  auto M_view = M->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto r_view = r->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto z_view = z->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  while (iter<maxiter && rnorm[0]/r0>tol) {
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyPrectimer);
      parallel_for("PCG apply prec",
                   RangePolicy<LA_exec>(0,z_view.extent(0)),
                   MRHYDE_LAMBDA (const int k ) {
        z_view(k,0) = r_view(k,0)/M_view(k,0);
      });
    }
    
    rho1 = rho;
    r->dot(*z, dotprod);
    rho = dotprod[0];
    if (iter == 0) {
      p->assign(*z);
    }
    else {
      beta = rho/rho1;
      p->update(one,*z,beta);
    }
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
      J->apply(*p,*q);
    }
    
    p->dot(*q,dotprod);
    pq = dotprod[0];
    alpha = rho/pq;
    
    x->update(alpha,*p,one);
    r->update(-one*alpha,*q,one);
    r->norm2(rnorm);
    
    iter++;
  }
  if (verbosity >= 10 && Comm->getRank() == 0) {
    cout << " ******* PCG Convergence Information: " << endl;
    cout << " *******     Iter: " << iter << "   " << "rnorm = " << rnorm[0]/r0 << endl;
  }
}

// ========================================================================================
// Specialized matrix-free PCG
// ========================================================================================

template<class Node>
void SolverManager<Node>::matrixFreePCG(const size_t & set, vector_RCP & b, vector_RCP & x,
                                        vector_RCP & M, const ScalarT & tol, const int & maxiter) {
  
  Teuchos::TimeMonitor localtimer(*PCGtimer);
  
  typedef typename Node::execution_space LA_exec;
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
  
  ScalarT rho = 1.0, rho1 = 0.0, alpha = 0.0, beta = 1.0, pq = 0.0;
  ScalarT one = 1.0, zero = 0.0;
  
  vector_RCP p, q, r, z, p_over, q_over;
  if (store_vectors) {
    p = p_pcg[set];
    q = q_pcg[set];
    r = r_pcg[set];
    z = z_pcg[set];
    p_over = p_pcg_over[set];
    q_over = q_pcg_over[set];
  }
  else {
    p = linalg->getNewVector(set);
    q = linalg->getNewVector(set);
    r = linalg->getNewVector(set);
    z = linalg->getNewVector(set);
    if (linalg->getHaveOverlapped()) {
      p_over = linalg->getNewOverlappedVector(set);
      q_over = linalg->getNewOverlappedVector(set);
    }
    else {
      p_over = p;
      q_over = q;
    }
  }
   
  p->putScalar(zero);
  q->putScalar(zero);
  r->putScalar(zero);
  z->putScalar(zero);
  
  if (linalg->getHaveOverlapped()) {
    p_over->putScalar(zero);
    q_over->putScalar(zero);
  }

  int iter=0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  
  {
    Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
    if (linalg->getHaveOverlapped()) {
      linalg->importVectorToOverlapped(set, p_over, x);
    }
    else {
      p_over->assign(*x);
    }
    assembler->applyMassMatrixFree(set, p_over, q_over);
    if (linalg->getHaveOverlapped()) {
      linalg->exportVectorFromOverlapped(set, q, q_over);
    }
  }
  
  r->assign(*b);
  r->update(-one,*q,one);
  
  // Fixed DOFs: zero r and M*p there, so the iteration only sees M_II and x keeps its fixed values
  auto fixed = explicitFixedDOF[set];
  auto r_view = r->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto q_view = q->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  parallel_for("PCG mask res",
               RangePolicy<LA_exec>(0,r_view.extent(0)),
               MRHYDE_LAMBDA (const int k ) {
    if (fixed(k)) {
      r_view(k,0) = 0.0;
    }
  });
  
  r->norm2(rnorm);
  ScalarT r0 = rnorm[0];
  
  auto M_view = M->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto z_view = z->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  while (iter<maxiter && rnorm[0]/r0>tol) {
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyPrectimer);
      parallel_for("PCG apply prec",
                   RangePolicy<LA_exec>(0,z_view.extent(0)),
                   MRHYDE_LAMBDA (const int k ) {
        z_view(k,0) = r_view(k,0)/M_view(k,0);
      });
    }
    
    rho1 = rho;
    r->dot(*z, dotprod);
    rho = dotprod[0];
    if (iter == 0) {
      p->assign(*z);
    }
    else {
      beta = rho/rho1;
      p->update(one,*z,beta);
    }
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, p_over, p);
      }
      q_over->putScalar(zero);
      assembler->applyMassMatrixFree(set, p_over, q_over);
      if (linalg->getHaveOverlapped()) {
        linalg->exportVectorFromOverlapped(set, q, q_over);
      }
      parallel_for("PCG mask Mp",
                   RangePolicy<LA_exec>(0,q_view.extent(0)),
                   MRHYDE_LAMBDA (const int k ) {
        if (fixed(k)) {
          q_view(k,0) = 0.0;
        }
      });
    }
    
    p->dot(*q,dotprod);
    pq = dotprod[0];
    alpha = rho/pq;
    
    x->update(alpha,*p,one);
    r->update(-one*alpha,*q,one);
    r->norm2(rnorm);
    
    iter++;
  }
  if (verbosity >= 10 && Comm->getRank() == 0) {
    cout << " ******* PCG Convergence Information: " << endl;
    cout << " *******     Iter: " << iter << "   " << "rnorm = " << rnorm[0]/r0 << endl;
  }
}
