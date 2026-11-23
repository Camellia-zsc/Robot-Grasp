# Graph Report - 智能车项目  (2026-11-24)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 5004 nodes · 14094 edges · 142 communities (95 shown, 47 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 761 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Teach Poses Components
- Scenarios Components
- Onsite Components
- Sim Camera Components
- Simulation World State
- Teach Classes Components
- Grasp Components
- Scenarios Components
- Preview Simulation Session
- MuJoCo Arm Control
- Browser Task Controls
- Placement Region Observation
- Segment Components
- Legacy Task Runner
- Pyagxarm Arm Components
- MuJoCo Scene Construction
- Legacy Task Runner
- Legacy Motion Skills
- Onsite Components
- Calib Verify Components
- Calib Points Components
- Teach Classes Components
- Ros Camera Components
- Color Calib Components
- Export Onnx Components
- Coordinate Transform Utilities
- Run Task Components
- Pipeline Components
- Calib Points Components
- Camera Configuration Geometry
- Language Model Planning
- Record Components
- Motion Ctl Components
- Standalone Cube Detection
- Observation Policy Components
- Composable Skill Orchestration
- Ui Components
- Validated Skill Execution
- Safety Components
- Web Service Routes
- Extrinsics Components
- Simulation World State
- Sim Arm Components
- Ik Components
- Interfaces Components
- Preview Simulation Session
- Make Dataset Components
- Run Components
- Selfcheck Components
- Camera Handeye Calibration
- Logger Components
- Geometry Components
- Render Components
- Sim Bench Components
- Obstacle Recovery Components
- Teach Record Components
- Interactive Physics Session
- Pyagxarm Arm Components
- Views Components
- Features Components
- Reach Components
- Skill Protocol Components
- Table Components
- Check Setup Components
- Cli Components
- Kinematics Components
- MuJoCo Physics Engine
- Classify Knn Components
- Camera Configuration Geometry
- Check Setup Components
- Operator Components
- Color Calib Components
- Color Calib Components
- Teach Record Components
- Browser Scene Editor
- Video Components
-  Components
- Stack Scene Observation
- Color Calib Components
- Assistant Bench Components
- Scenarios Components
- Video Components
- Detector Onnx Components
- Detector Onnx Components
- Live Components
- Preview Simulation Session
- Selfcheck Components
- Recorder Components
- Grasp Components
- Standalone Cube Detection
- Catalog Perception Components
- Factory Components
- Check Onnx Components
- Live Components
- Clock Components
- L1 New Components
- Camera Configuration Geometry
- Detector Onnx Components
- Selection Components
- Calib Points Components
- Detector Onnx Components
- MuJoCo Scene Construction
- Calib Points Components
- Color Calib Components
- Pyagxarm Arm Components
- Web Service Routes
- Recovery Components
- Legacy Task Runner
- Color Calib Components
- Color Calib Components
- Dialogue Context Management
- Preview Simulation Session
- Recorder Components
- Pyagxarm Arm Components
- Run Components
- Sim Bench Components
- Sim Bench Components
- Common Components
- Teach Poses Components
- Task Contract Components
- Utils Components
- Goal Readiness Components
- Moveit Bridge Components
- Simulation World State
- Sim Arm Components
- MuJoCo Arm Control
- Color Calib Components
- Detector Onnx Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 117 edges
2. `SimulationSession` - 92 edges
3. `TaskRunner` - 82 edges
4. `SimObject` - 79 edges
5. `CameraIntrinsics` - 67 edges
6. `MujocoArm` - 62 edges
7. `Frame` - 60 edges
8. `inv_T()` - 55 edges
9. `PyAgxArm` - 51 edges
10. `SafeArm` - 50 edges

## Surprising Connections (you probably didn't know these)
- `measured_neighbors()` --calls--> `stationary()`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/active_observation.py → ws/src/cube_task/cube_task/assistant/skill_execution.py
- `_training_samples()` --uses--> `CameraIntrinsics`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/core/types.py
- `make_sandbox_perception()` --uses--> `Perception`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/perception/pipeline.py
- `MotionObservation` --uses--> `Detection`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/motion_observation.py → ws/src/cube_task/cube_task/core/types.py
- `DashboardPhysics` --uses--> `Config`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/core/config.py

## Import Cycles
- None detected.

## Communities (142 total, 47 thin omitted)

### Community 0 - "Teach Poses Components"
Cohesion: 0.02
Nodes (64): Config, deep_merge(), _find_root(), load_config(), package_root(), _parse_value(), read_yaml(), _set_dotted() (+56 more)

### Community 1 - "Scenarios Components"
Cohesion: 0.03
Nodes (55): g(), object_mass(), _object_parts(), _arc_capsules(), _as_list(), _axis_angle(), _banana_parts(), _catalog_object() (+47 more)

### Community 2 - "Onsite Components"
Cohesion: 0.03
Nodes (47): ArmError, build_parser(), check_args(), main(), build_parser(), check_args(), main(), auto_due() (+39 more)

### Community 3 - "Sim Camera Components"
Cohesion: 0.04
Nodes (23): _make_sim_camera(), _pose(), _check_intr(), _has_dist(), register_depth_to_color(), fit_intrinsics(), CameraSource, Clock (+15 more)

### Community 4 - "Simulation World State"
Cohesion: 0.03
Nodes (18): LivePreview, _release_xys(), _box_surface(), _capsule_surface(), _check_color(), check_pose(), _cylinder_surface(), densify_polyline() (+10 more)

### Community 5 - "Teach Classes Components"
Cohesion: 0.04
Nodes (37): accepted_labels(), add_sim_ground_truth(), Batch, build_report(), clean_label(), config_hints(), confirm_labels(), confused_pairs() (+29 more)

### Community 6 - "Grasp Components"
Cohesion: 0.05
Nodes (28): choose_closing_angle(), neutral_closing_angle(), wrap_pi(), Zone, capsule_edge_margin(), _finite(), footprint_capsule(), GraspPlanner (+20 more)

### Community 7 - "Scenarios Components"
Cohesion: 0.04
Nodes (46): completed_position(), _in_completed_zone(), matching_objects(), _compound_core_supported(), mask_carried_depth(), _pose(), _solid_box_source(), _source_surface_supported() (+38 more)

### Community 9 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (7): GripperResult, MotionResult, _cfg_float(), _cfg_vec(), MujocoArm, _Profile, _Trajectory

### Community 10 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 11 - "Placement Region Observation"
Cohesion: 0.05
Nodes (36): _coverage(), inspect_held(), measured_neighbors(), measurements_changed(), candidate_visibility_filter(), allowed(), _dimensions(), forget_inspection() (+28 more)

### Community 12 - "Segment Components"
Cohesion: 0.06
Nodes (33): _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges(), depth_spikes() (+25 more)

### Community 15 - "MuJoCo Scene Construction"
Cohesion: 0.07
Nodes (34): _add_environment(), _add_object(), _add_zone(), _as_list(), assets_dir_for(), build_scene(), _camera(), _camera_housing() (+26 more)

### Community 16 - "Legacy Task Runner"
Cohesion: 0.06
Nodes (24): support_bottom(), center_supported(), detection_table_z(), held_offset_of(), transit_height(), polygons_overlap(), prepare_polygon(), PreparedPolygon (+16 more)

### Community 17 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (12): GraspPlan, PickResult, PlacePlan, _force_value(), gripper_force_for(), is_fault_motion_error(), _Lost, PickPlaceSkill (+4 more)

### Community 18 - "Onsite Components"
Cohesion: 0.05
Nodes (27): aruco_dictionary(), correction_formula(), arm_firmware_version(), ArmPose, average_poses(), backup_file(), backup_path(), dotted_keys() (+19 more)

### Community 19 - "Calib Verify Components"
Cohesion: 0.05
Nodes (19): corrected_extrinsics(), _fmt_d(), HoverPlan, measure_help(), parent_pose_in_base(), plan_text(), Sample, samples_table() (+11 more)

### Community 20 - "Calib Points Components"
Cohesion: 0.05
Nodes (13): average_measurements(), _Calib, draw_measurement(), iso_now(), Measurement, measurement_line(), pairs_from_json(), pairs_to_json() (+5 more)

### Community 21 - "Teach Classes Components"
Cohesion: 0.05
Nodes (10): format_table(), teach_hint(), command_kind(), counts_text(), instance_rows(), layout_unchanged(), plan_labeling(), _Teacher (+2 more)

### Community 22 - "Ros Camera Components"
Cohesion: 0.06
Nodes (9): FileCamera, camera_info_to_intrinsics(), check_depth_scale(), depth_msg_to_meters(), image_msg_to_bgr(), _image_view(), RosCamera, _stamp_to_sec() (+1 more)

### Community 23 - "Color Calib Components"
Cohesion: 0.04
Nodes (24): parse_measurement(), build_parser(), Command, main(), match_label(), normalize_line(), options_from_args(), parse_command() (+16 more)

### Community 24 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 25 - "Coordinate Transform Utilities"
Cohesion: 0.06
Nodes (32): closing_angle_of(), fit_rigid(), make_T(), pose6_to_T(), quat_to_rot(), rot_to_quat(), rot_to_rpy(), rot_x() (+24 more)

### Community 26 - "Run Task Components"
Cohesion: 0.07
Nodes (33): RealClock, main(), motion_tuning_file(), _open(), is_teaching(), _arm_text(), _attach_mujoco(), _base_summary() (+25 more)

### Community 27 - "Pipeline Components"
Cohesion: 0.09
Nodes (12): Detector2D, InstanceClassifier, Detection, Frame, PerceptionResult, ColorClassifier, _finite(), mask_in_box_overlap() (+4 more)

### Community 28 - "Calib Points Components"
Cohesion: 0.05
Nodes (24): _as_array(), _as_list(), compare_extrinsics(), _copy_or_none(), _deg_between(), detect_markers(), find_outliers(), _lsq_plane() (+16 more)

### Community 29 - "Camera Configuration Geometry"
Cohesion: 0.07
Nodes (21): _as_matrix(), camera_pose_in_base(), _check_rigid(), Extrinsics, extrinsics_from_dict(), is_optical_frame(), _matrix_list(), _parse_transform() (+13 more)

### Community 30 - "Language Model Planning"
Cohesion: 0.06
Nodes (16): _CancellableStream, _iter_sse(), _json_object(), LLMError, LLMSettings, load_settings(), _model_placement(), number() (+8 more)

### Community 31 - "Record Components"
Cohesion: 0.07
Nodes (19): CameraConfig, cmd_record(), _collect_live(), ArmPoseReader, _image_to_numpy(), RosRgbd, _atomic_imwrite(), _atomic_text() (+11 more)

### Community 32 - "Motion Ctl Components"
Cohesion: 0.07
Nodes (23): MoveItBridge, observe_views(), _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick() (+15 more)

### Community 33 - "Standalone Cube Detection"
Cohesion: 0.07
Nodes (13): Intrinsics, color_mask(), CubeDetection, depth_to_base(), draw_result(), fit_table_plane(), _in_polygon(), L1Detector (+5 more)

### Community 34 - "Observation Policy Components"
Cohesion: 0.07
Nodes (17): CarriedDetections, restoration_yaw_period(), MotionObservation, measurement_absent(), measurement_visible(), merge_local_detections(), rank_views(), _same_layer() (+9 more)

### Community 36 - "Ui Components"
Cohesion: 0.05
Nodes (11): camera_pose_for_frame(), compose_view(), depth_range_auto(), draw_numbers(), recorded_flange_pose(), text_panel(), depth_colormap(), display_available() (+3 more)

### Community 37 - "Validated Skill Execution"
Cohesion: 0.10
Nodes (8): _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search(), _TargetUnavailable

### Community 38 - "Safety Components"
Cohesion: 0.08
Nodes (3): _rejected(), SafeArm, SafetyGuard

### Community 39 - "Web Service Routes"
Cohesion: 0.09
Nodes (10): idle_batch(), BusyError, DashboardController, editable_session(), run(), work(), run(), run() (+2 more)

### Community 40 - "Extrinsics Components"
Cohesion: 0.09
Nodes (20): _check_radians(), Extrinsics, extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml(), _link_to_optical(), load_extrinsics(), non_optical_frame_message() (+12 more)

### Community 43 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 45 - "Preview Simulation Session"
Cohesion: 0.06
Nodes (14): whole_footprint_at_goal(), whole_footprint_in_zone(), catalog_entries(), CatalogEditor, _segment_in_rectangle(), completion_outline(), destination_label(), footprint_in_destination() (+6 more)

### Community 46 - "Make Dataset Components"
Cohesion: 0.08
Nodes (17): build_dataset(), build_parser(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes (+9 more)

### Community 47 - "Run Components"
Cohesion: 0.07
Nodes (10): _EventTap, expected_zones(), failure_causes(), final_truth(), _object_cause(), _ObjState, PhysicsMonitor, _pick_cause() (+2 more)

### Community 48 - "Selfcheck Components"
Cohesion: 0.09
Nodes (26): arm_self_check(), at_observe_pose(), _check_arm(), _check_assets(), _check_camera(), _check_extrinsics(), _check_reachable(), _check_robot_config() (+18 more)

### Community 49 - "Camera Handeye Calibration"
Cohesion: 0.08
Nodes (14): cmd_board(), cmd_handeye(), board_corners_3d(), BoardSpec, coverage_warning(), detect_board(), _dictionary(), draw_detection() (+6 more)

### Community 50 - "Logger Components"
Cohesion: 0.09
Nodes (5): dumps_compact(), RunLogger, safe_filename(), _short(), to_jsonable()

### Community 51 - "Geometry Components"
Cohesion: 0.09
Nodes (19): as_zone_list(), backproject(), check_depth(), check_intrinsics(), densify_polyline(), _dist_array(), has_distortion(), _intrinsics_key() (+11 more)

### Community 52 - "Render Components"
Cohesion: 0.09
Nodes (19): _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box(), _hit_capsule(), _hit_cylinder() (+11 more)

### Community 53 - "Sim Bench Components"
Cohesion: 0.09
Nodes (21): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), format_report(), ground_truth(), is_success() (+13 more)

### Community 54 - "Obstacle Recovery Components"
Cohesion: 0.11
Nodes (26): _action(), _ActionFailed, _center(), _destination(), execute_held_recovery(), record(), stopped(), _fresh_target() (+18 more)

### Community 55 - "Teach Record Components"
Cohesion: 0.09
Nodes (15): atomic_write_text(), contact_sheet(), crop_of(), instance_records(), load_instances(), load_labels(), load_samples(), loo_predictions() (+7 more)

### Community 56 - "Interactive Physics Session"
Cohesion: 0.08
Nodes (6): configure_sandbox(), make_sandbox_perception(), placement_metadata(), DashboardPhysics, _Cancellation, _ViewerClock

### Community 57 - "Pyagxarm Arm Components"
Cohesion: 0.09
Nodes (3): _import_sdk(), SdkCallError, _SdkPort

### Community 58 - "Views Components"
Cohesion: 0.09
Nodes (18): _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform(), coverage_report() (+10 more)

### Community 59 - "Features Components"
Cohesion: 0.09
Nodes (13): _check_ranges(), hsv_ranges_from_samples(), in_ranges(), bgr_to_hsv_pixels(), _crop_mask(), feature_vector(), footprint_shape(), hue_histogram() (+5 more)

### Community 60 - "Reach Components"
Cohesion: 0.12
Nodes (17): measured_tapered_contact(), supported_grasp(), gap(), accepts_kwarg(), adjust_plan_poses(), _at_z(), _check(), free_tilt_limit() (+9 more)

### Community 61 - "Skill Protocol Components"
Cohesion: 0.15
Nodes (22): _number(), validate_destination(), plan_limit(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _final_audit_summary() (+14 more)

### Community 62 - "Table Components"
Cohesion: 0.10
Nodes (8): TablePlane, parse_polygon(), fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), _restrict(), TableFit, TableParams

### Community 63 - "Check Setup Components"
Cohesion: 0.10
Nodes (13): main(), build_parser(), camera_topics(), can_up_command(), check_can(), check_modules(), check_ros(), main() (+5 more)

### Community 64 - "Cli Components"
Cohesion: 0.12
Nodes (12): build_parser(), cmd_l1(), cmd_show(), cmd_transform(), _collect_offline(), _default(), _load_cam(), main() (+4 more)

### Community 65 - "Kinematics Components"
Cohesion: 0.15
Nodes (3): _as_joints(), _check_pose(), PiperKinematics

### Community 69 - "Check Setup Components"
Cohesion: 0.11
Nodes (7): check_config(), check_live(), config_fix(), config_messages(), Item, _live_arm(), Report

### Community 71 - "Color Calib Components"
Cohesion: 0.11
Nodes (11): effective_classes(), fmt_ranges(), inside_fraction(), missing_label_warnings(), overlap_fractions(), overlap_pairs(), overlap_warnings(), ranges_array() (+3 more)

### Community 72 - "Color Calib Components"
Cohesion: 0.11
Nodes (3): calib_header(), InstanceSample, SampleBook

### Community 73 - "Teach Record Components"
Cohesion: 0.13
Nodes (5): check_label(), parse_review_command(), parse_sample_id(), Reviewer, suspects()

### Community 74 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 76 - " Components"
Cohesion: 0.12
Nodes (8): bind_item(), build_items(), validate_decision(), valid_zone(), _xyz_is_finite(), _dashboard_event(), DashboardServer, main()

### Community 77 - "Stack Scene Observation"
Cohesion: 0.13
Nodes (9): _box_side_surfaces(), _LayerPerception, _observed_levels(), refine_supported_objects(), _side_support_evidence(), _support_evidence(), transform_cloud(), rect_geometry() (+1 more)

### Community 79 - "Assistant Bench Components"
Cohesion: 0.16
Nodes (12): camera_pose_in_base(), BenchSession, record_invalidation(), record_release(), main(), random_objects(), run_case(), session_factory() (+4 more)

### Community 80 - "Scenarios Components"
Cohesion: 0.13
Nodes (9): _cache_path(), _classifier(), _training_samples(), layout_bottom_px(), layout_intrinsics(), look_at(), official_visible(), seen() (+1 more)

### Community 81 - "Video Components"
Cohesion: 0.15
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 82 - "Detector Onnx Components"
Cohesion: 0.16
Nodes (7): _check_image(), decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), unletterbox_boxes()

### Community 83 - "Detector Onnx Components"
Cohesion: 0.14
Nodes (5): _first_line(), OnnxInfo, _onnxruntime_available(), _OpenCVRunner, _OrtRunner

### Community 84 - "Live Components"
Cohesion: 0.16
Nodes (6): apply_live(), drop_live_seen(), drop_unconfirmed_live(), LiveSettings, LiveWatcher, _xy()

### Community 86 - "Selfcheck Components"
Cohesion: 0.15
Nodes (9): make_camera(), extrinsics_from_cfg(), _camera_frame(), camera_pose_for_check(), check_pick_area(), _pick_area_depth(), pick_area_mask(), _live_camera() (+1 more)

### Community 87 - "Recorder Components"
Cohesion: 0.20
Nodes (7): _check_idx(), depth_to_mm(), load_frame(), _name(), save_frame(), median_depth(), valid_depth_mask()

### Community 88 - "Grasp Components"
Cohesion: 0.15
Nodes (6): parse_zone(), parse_zones(), expected_table_z(), _float(), max_gripper_width(), _required_float()

### Community 89 - "Standalone Cube Detection"
Cohesion: 0.15
Nodes (4): _import_piper_vision(), _load_l1(), _check_ranges(), L1Params

### Community 90 - "Catalog Perception Components"
Cohesion: 0.16
Nodes (5): _center_round_fruit(), _ColorSeededKnn, completeness(), pear_pose(), SandboxPerception

### Community 91 - "Factory Components"
Cohesion: 0.17
Nodes (6): classifier_backend(), make_classifier(), make_detector(), make_perception(), task_labels(), _knn_loadable()

### Community 92 - "Check Onnx Components"
Cohesion: 0.17
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 94 - "Clock Components"
Cohesion: 0.15
Nodes (4): SimClock, make_arm(), kinematic_arm(), kinematic_arm()

### Community 95 - "L1 New Components"
Cohesion: 0.19
Nodes (3): points_in_zones(), L1NewPerception, annotate_neighbors()

### Community 96 - "Camera Configuration Geometry"
Cohesion: 0.18
Nodes (7): _apply(), _atomic_write(), load_camera_config(), points_base_to_camera(), points_camera_to_base(), save_camera_config(), write_extrinsics()

### Community 97 - "Detector Onnx Components"
Cohesion: 0.26
Nodes (7): build_label_mapping(), _check_metadata(), _check_names(), DetectorError, _probability(), read_labels(), _resolve_input_size()

### Community 98 - "Selection Components"
Cohesion: 0.25
Nodes (8): clearances(), is_blacklisted(), _radius(), select_target(), dist(), iso_key(), tie(), _xy()

### Community 99 - "Calib Points Components"
Cohesion: 0.18
Nodes (5): BlockParams, clean_depth(), fit_plane_ransac(), measure_target_aruco(), measure_target_block()

### Community 100 - "Detector Onnx Components"
Cohesion: 0.27
Nodes (6): _fields(), opencv_version(), read_onnx_info(), _text(), _value_info(), _varint()

### Community 101 - "MuJoCo Scene Construction"
Cohesion: 0.17
Nodes (5): _cfg_get(), require_mujoco(), _geom_min_z(), intrinsics_for_fovy(), piper_geometry()

### Community 104 - "Color Calib Components"
Cohesion: 0.18
Nodes (3): draw_texts(), hue_distance(), preview_rows()

### Community 107 - "Recovery Components"
Cohesion: 0.39
Nodes (5): _copy_fields(), model_recovery_context(), _number(), _text(), validate_recovery_decision()

### Community 109 - "Color Calib Components"
Cohesion: 0.22
Nodes (4): build_colors_yaml(), _FlowList, overridden_classes(), plain_ranges()

### Community 110 - "Color Calib Components"
Cohesion: 0.22
Nodes (3): hue_stats(), _num(), view_rows()

### Community 111 - "Dialogue Context Management"
Cohesion: 0.32
Nodes (3): conversation_summary(), model_conversation(), _safe_text()

### Community 113 - "Recorder Components"
Cohesion: 0.29
Nodes (3): _imwrite_atomic(), list_frames(), next_index()

### Community 114 - "Pyagxarm Arm Components"
Cohesion: 0.25
Nodes (4): piper_firmware_tuple(), seamless_teach_switch(), teach_exit_hint(), _arm_fix()

### Community 120 - "Task Contract Components"
Cohesion: 0.47
Nodes (3): _all_default_request(), bind_task_contract(), _inventory_labels()

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 2054 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **47 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_knn_from_sim()` connect `Teach Poses Components` to `Classify Knn Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `SimulationSession` (e.g. with `CarriedDetections` and `CatalogEditor`) actually correct?**
  _`SimulationSession` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Teach Poses Components` be split into smaller, more focused modules?**
  _Cohesion score 0.02117622929805671 - nodes in this community are weakly interconnected._
- **Why does `collect_sim_samples()` connect `Teach Poses Components` to `Scenarios Components`, `Pipeline Components`, `Assistant Bench Components`, `Selfcheck Components`, `Features Components`, `Clock Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Should `Scenarios Components` be split into smaller, more focused modules?**
  _Cohesion score 0.030752026838132513 - nodes in this community are weakly interconnected._
- **Why does `SimObject` connect `Scenarios Components` to `Simulation World State`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1420 raw AST edges have unresolved endpoints; 6 self-loops; 774 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
