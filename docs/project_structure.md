# Project structure

```text
biosensor_priors/
    configs/
        pipeline.yaml
        fitness.yaml
        search.yaml
        thresholds.yaml
        structures.yaml
        physics.yaml
        rf3_physics.yaml
        ablation.yaml

    data/
        experimental/
        constructs/
        ligands/
        structures/
        physics/
        rounds/

    docs/                    # Sphinx + Read the Docs
        conf.py
        index.md
        stages/

    src/biosensor_priors/
        common/
            schemas.py
            identifiers.py
            provenance.py
            canonical.py
            gates.py
            config.py
            ipsae.py

        stage0_ground_truth/
            load_experiments.py
            fitness.py
            splits.py
            validate.py

        stage1_structures/
            make_jobs.py
            adapters/
            confidence.py
            structural_compare.py
            ipsae_compare.py
            gate1.py

        stage2_physics/
            ligand_ensemble.py
            rif_jobs.py
            mutation_scan.py
            score_parser.py
            physics_uncertainty.py
            gate2.py
            wrappers/run_rf3_dock.py

        stage3_surrogate/
            features.py
            physics_mean.py
            construct_intercept.py
            confidence_weighting.py
            kernels.py
            gp_residual.py
            phenotypes.py
            calibration.py
            surrogate.py
            cross_validate.py
            gate3.py

        stage4_search/
            design_space.py
            prefilter.py
            policy.py
            random_search.py
            adalead.py
            mcmc.py
            bo.py
            thompson.py
            bo_evo.py
            acquisition.py
            batch_design.py

        stage5_prospective/
            freeze_predictions.py
            import_results.py
            prospective_validation.py
            update_model.py
            gate4.py

        stage6_ablation/
            experiments.py
            statistics.py
            figures.py
            report.py

    tests/
        test_numbering.py
        test_Q324R.py
        test_A355R.py
        test_no_data_leakage.py
        test_reproducibility.py
        test_ml_upgrades.py

    manifests/
    outputs/
```

Package import root: ``biosensor_priors`` (src layout). Configuration and data
remain outside the package so they can version independently of code releases.
