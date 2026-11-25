# Packaging verification

Checks executed on Windows with the existing local Python 3.14 runtime. No models were trained and no historical scientific values were changed.

| Check | Status | Evidence / scope |
|---|---|---|
| python_syntax | PASS | 13 files compiled in memory; no bytecode generated |
| notebook_integrity | PASS | 11 notebooks; 59 Python cells parsed; 0 shell/magic cells inspected separately; no rich media |
| original_fitting_code_identity | PASS | All code cells in 7 portable notebooks unchanged from verified archive |
| frozen_artifact_and_split_identity | PASS | Original workflow, preprocessing, Task A runner, selected weights and partitions unchanged byte-for-byte |
| relative_configuration | PASS | exit=0; 0.2s; transcript outside public repository |
| public_upload_only_startup | RESOLVED | Initial test harness defaulted to 3 seconds; relocated public-only recheck passed with 120-second allowance. No product-code fix was required. |
| encoder_offline_setup | PASS | exit=0; 0.4s; transcript outside public repository |
| private_demo_preparation | PASS | exit=0; 0.3s; transcript outside public repository |
| real_inference_and_incident_workflow | PASS | exit=0; 19.0s; transcript outside public repository |
| task_b_fresh_frozen_evaluation | PASS | exit=0; 1.6s; transcript outside public repository |
| task_b_private_dataset_hashes | PASS | exit=0; 0.4s; transcript outside public repository |
| task_a_dataset_hashes_all_500 | PASS | exit=0; 8.6s; transcript outside public repository |
| task_a_fresh_four_clip_check | PASS | exit=0; 31.6s; transcript outside public repository |
| threshold_boundaries | PASS | exit=0; 3.5s; transcript outside public repository |
| source_archive_unchanged | PASS | Original verified archive checksum still matches |
| full_holdout_rerun | NOT RUN | Not repeated: prior full 100-video verification retained; short fresh check is separate |
| model_training | NOT RUN | Preserved original fitting source; no SVM/MLP/GRU training performed |
| clean_environment_installation | NOT RUN | Uses documented existing local dependencies and ignored test-only torchvision; fresh pip installation not demonstrated |
| official_dataset_download | NOT RUN | Official README and Drive partition links checked; full online snapshot download not repeated |
| public_upload_only_startup_recheck | PASS | Relocated public-only checkout; 120-second cold-import allowance; no media or weights supplied |
| recovered_functions_dumpwatch_task_b_final_frozen.ipynb | PASS | 18 original functions/classes checked; missing=[]; changed=[] |
| recovered_functions_dumpwatch_task_b_exploratory.ipynb | PASS | 19 original functions/classes checked; missing=[]; changed=[] |
| recovered_functions_dumpwatch_task_b_local_comparison.ipynb | PASS | 27 original functions/classes checked; missing=[]; changed=[] |
| streamlit_loopback_health | PASS | HTTP health responded 200 |
| nbformat_schema_validation | PASS | 11 final notebooks validated with nbformat; no files modified. |

## Interpretation

Fresh Task A check decoded four original holdout videos and compared three frozen methods against saved scores, classifications and onsets. It separately recalculated archived 100-video metrics. The prior full fresh 100-video verification is retained under QA/Validation/Completion; it was not rerun during packaging.
Task B freshly evaluated eight development and nine retrospective test pairs with the saved SVM, saved MLP, image difference and majority baseline. All saved decisions/scores and checked metrics matched. The retrospective design remains a limitation.
Real application tests exercised positive/negative videos, remains/review/clear-photo pairs, registration guards, SQLite history, blocked unchecked closure and explicit human-approved closure. Databases were separate temporary files. A real local Streamlit server returned HTTP 200 from its health endpoint.
The original bundled torchvision distribution was copied into ignored local Demo/vendor for these Windows tests. The Task A check additionally used the existing archived sklearn 1.9.1 runtime (original pickle 1.6.1); it emitted InconsistentVersionWarning while matching the four locked clips. Those dependencies are NOT included in the public archive. These are relocated-package checks using an existing environment, not proof of a fresh pip installation or general cross-version pickle compatibility.
The encoder setup was exercised with the already verified official checkpoint using --from-file. Online model download and complete independent online MIVIA download were not repeated. Private datasets were read locally and never included in publication.
11 notebook files and 59 Python code cells were parsed. The portable fitting notebook code cells match the verified source archive. Scientific AST definitions also match recovered originals: selected SVM 18, original synthetic comparison 19, local comparison 27. Historical executed companions retain numeric/text output but no image output.
Logs with personal machine paths remain outside the public repository. A local-source archive with SHA-256 01ca97a7b2a019225efc047a447c719a0284a4001c9d17ed0271d094f4084c73 was preserved and is unchanged.
Before publication, PUBLICATION_SHA256.json and tools/verify_package.py validate exact staged files, privacy exclusions and hash preservation. The clean ZIP is built only from this manifest, not from ignored test assets. Git line-ending conversion is disabled so frozen split/config hashes survive clone and extraction.
Full training reproducibility remains NOT DEMONSTRATED: original source and settings are available, but fitting and clean GPU/environment installation were deliberately not repeated. Dataset access for the instructor still requires the private Task B email handoff and MIVIA hash validation after download.
