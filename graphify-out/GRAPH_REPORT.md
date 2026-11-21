# Graph Report - 智能车项目  (2026-11-21)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4796 nodes · 13368 edges · 149 communities (107 shown, 42 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 712 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Teach Classes Components
- Obstacle Recovery Components
- Onsite Components
- Safety Components
- Scenarios Components
- Teach Record Components
- Kinematics Components
- Color Calib Components
- Legacy Task Runner
- Browser Task Controls
- MuJoCo Arm Control
- Geometry Components
- Simulation World State
- Segment Components
- Pyagxarm Arm Components
- Teach Poses Components
- Color Calib Components
- Calib Verify Components
- Teach Classes Components
- Legacy Task Runner
- Ros Camera Components
- Legacy Motion Skills
- Onsite Components
- Standalone Cube Detection
- Sim Arm Components
- Sim Bench Components
- Config Components
- Camera Configuration Geometry
- Extrinsics Components
- Teach Poses Components
- Grasp Components
- Simulation World State
- Scenarios Components
- Run Task Components
- Selfcheck Components
- Web Service Routes
- Ik Components
- Check Setup Components
- Camera Handeye Calibration
- Preview Simulation Session
- Placement Region Observation
- Make Dataset Components
- Validated Skill Execution
- Logger Components
- Record Components
- Stack Scene Observation
- Pipeline Components
- Live Components
- Export Onnx Components
- Pyagxarm Arm Components
- Coordinate Transform Utilities
- Render Components
- Classify Knn Components
- Grasp Components
- Motion Ctl Components
- Features Components
- Sim Camera Components
- Composable Skill Orchestration
- Simulation World State
- Preview Simulation Session
- Calib Points Components
- Onsite Components
- MuJoCo Physics Engine
- Views Components
- Calib Points Components
- Factory Components
- Ui Components
- Preview Simulation Session
- Language Model Planning
- Live Components
- MuJoCo Scene Construction
- Operator Components
- Calib Points Components
- Camera Configuration Geometry
- Recovery Components
- Teach Poses Components
- Calib Points Components
- Browser Scene Editor
- Train Yolo Components
- Catalog Perception Components
- Skill Protocol Components
- Table Components
- Video Components
- MuJoCo Scene Construction
- Interactive Physics Session
- Scenarios Components
- Moveit Bridge Components
- MuJoCo Scene Construction
- Run Components
- Teach Poses Components
- Detector Onnx Components
- Cli Components
- Detector Onnx Components
- Calib Points Components
- Calib Points Components
- Check Onnx Components
- Run Components
- Camera Configuration Geometry
- Preview Simulation Session
- Grasp Components
- Video Components
- Catalog Components
- Detector Onnx Components
- Visualize Components
- Check Setup Components
- Standalone Cube Detection
- Detector Onnx Components
- Observations Components
- Preview Simulation Session
- Teach Record Components
- MuJoCo Scene Construction
- MuJoCo Scene Construction
- Calib Points Components
- Language Model Planning
- Teach Poses Components
- Pyagxarm Arm Components
- Motion Ctl Components
- Web Service Routes
- Run Components
- Dialogue Context Management
- Run Components
- Calib Points Components
- Check Setup Components
- Run Task Components
- Onsite Components
- Frames Components
- Assistant Ui Components
- Language Model Planning
- Common Components
- Teach Poses Components
-   Main   Components
- Preview Simulation Session
- Sim Arm Components
- MuJoCo Arm Control
- MuJoCo Scene Construction
- Detector Onnx Components
- Scenarios Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 110 edges
2. `SimObject` - 79 edges
3. `TaskRunner` - 79 edges
4. `SimulationSession` - 73 edges
5. `CameraIntrinsics` - 67 edges
6. `MujocoArm` - 62 edges
7. `Frame` - 60 edges
8. `PyAgxArm` - 51 edges
9. `SafeArm` - 50 edges
10. `Perception` - 49 edges

## Surprising Connections (you probably didn't know these)
- `inspect_held()` --indirect_call--> `safe_inspection_route()`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/active_observation.py → ws/src/cube_task/cube_task/assistant/placement_observation.py
- `_training_samples()` --uses--> `CameraIntrinsics`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/core/types.py
- `make_sandbox_perception()` --uses--> `Perception`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/perception/pipeline.py
- `DashboardPhysics` --uses--> `Config`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/core/config.py
- `DashboardPhysics` --uses--> `SafeArm`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/robot/safety.py

## Import Cycles
- None detected.

## Communities (149 total, 42 thin omitted)

### Community 0 - "Teach Classes Components"
Cohesion: 0.03
Nodes (40): task_labels(), load_extrinsics_or_none(), accepted_labels(), add_sim_ground_truth(), Batch, build_report(), check_label(), clean_label() (+32 more)

### Community 1 - "Obstacle Recovery Components"
Cohesion: 0.04
Nodes (53): inspect_held(), bind_item(), build_items(), completed_position(), _in_completed_zone(), matching_objects(), whole_footprint_at_goal(), whole_footprint_in_zone() (+45 more)

### Community 2 - "Onsite Components"
Cohesion: 0.04
Nodes (41): next_index(), ArmError, build_parser(), main(), build_parser(), check_args(), main(), auto_due() (+33 more)

### Community 3 - "Safety Components"
Cohesion: 0.03
Nodes (5): ArmInterface, ArmHealth, _rejected(), SafeArm, SafetyGuard

### Community 4 - "Scenarios Components"
Cohesion: 0.05
Nodes (32): _as_list(), _axis_angle(), _catalog_object(), _catalog_objects(), _check_graspable(), _class_grasp(), compound_sections(), _edge_normals() (+24 more)

### Community 5 - "Teach Record Components"
Cohesion: 0.04
Nodes (25): _check_idx(), depth_to_mm(), _imwrite_atomic(), list_frames(), load_frame(), _name(), save_frame(), atomic_write_text() (+17 more)

### Community 6 - "Kinematics Components"
Cohesion: 0.05
Nodes (24): Clock, inv_T(), is_rotation(), pose_error(), make_arm(), _as_joints(), _check_pose(), flange_to_tcp_from_cfg() (+16 more)

### Community 7 - "Color Calib Components"
Cohesion: 0.04
Nodes (31): build_colors_yaml(), calib_header(), circular_mean_hue(), Command, compute_ranges(), _FlowList, fmt_ranges(), hue_stats() (+23 more)

### Community 8 - "Legacy Task Runner"
Cohesion: 0.05
Nodes (27): overlaps(), center_supported(), Detection, object_footprint(), object_prism(), annotate_grasp_neighbors(), detection_table_z(), held_offset_of() (+19 more)

### Community 9 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 10 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (6): MotionResult, _cfg_float(), _cfg_vec(), MujocoArm, _Profile, _Trajectory

### Community 11 - "Geometry Components"
Cohesion: 0.06
Nodes (30): _coverage(), measurements_changed(), footprint_in_destination(), rank_views(), _check_intr(), _has_dist(), register_depth_to_color(), camera_info_to_intrinsics() (+22 more)

### Community 12 - "Simulation World State"
Cohesion: 0.05
Nodes (9): LivePreview, _check_color(), check_pose(), _finite_array(), _normalize_part(), point_in_polygon(), _positive(), SimWorld (+1 more)

### Community 13 - "Segment Components"
Cohesion: 0.05
Nodes (37): _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges(), depth_spikes() (+29 more)

### Community 14 - "Pyagxarm Arm Components"
Cohesion: 0.06
Nodes (3): Stamped, PyAgxArm, zero_pose_problem()

### Community 15 - "Teach Poses Components"
Cohesion: 0.05
Nodes (33): parse_zone(), read_yaml_or_empty(), apply_changes(), ccw_polygon(), distance_to_polygon(), effective(), _finite_number(), _fmt_joints() (+25 more)

### Community 16 - "Color Calib Components"
Cohesion: 0.04
Nodes (11): _ColorCalib, draw_texts(), effective_classes(), hue_distance(), overlap_pairs(), overlap_warnings(), same_layout(), _View (+3 more)

### Community 17 - "Calib Verify Components"
Cohesion: 0.04
Nodes (23): approach_z(), corrected_extrinsics(), corrected_transform(), _deviations(), _fmt_d(), hover_plan(), HoverPlan, _mean3() (+15 more)

### Community 18 - "Teach Classes Components"
Cohesion: 0.04
Nodes (11): build_extra(), format_table(), read_pose(), command_kind(), counts_text(), instance_rows(), layout_unchanged(), parse_label_command() (+3 more)

### Community 19 - "Legacy Task Runner"
Cohesion: 0.07
Nodes (3): TaskReport, _NullLogger, TaskRunner

### Community 20 - "Ros Camera Components"
Cohesion: 0.06
Nodes (9): FileCamera, check_depth_scale(), depth_msg_to_meters(), fit_intrinsics(), image_msg_to_bgr(), _image_view(), RosCamera, _stamp_to_sec() (+1 more)

### Community 21 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (14): rotation_angle(), GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), _force_value(), gripper_force_for(), is_fault_motion_error() (+6 more)

### Community 22 - "Onsite Components"
Cohesion: 0.05
Nodes (25): correction_formula(), arm_firmware_version(), ArmPose, average_poses(), backup_file(), backup_path(), detection_rows(), ensure_motion_ready() (+17 more)

### Community 23 - "Standalone Cube Detection"
Cohesion: 0.07
Nodes (15): _import_piper_vision(), Intrinsics, cmd_l1(), color_mask(), CubeDetection, depth_to_base(), draw_result(), fit_table_plane() (+7 more)

### Community 25 - "Sim Bench Components"
Cohesion: 0.06
Nodes (23): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), FaultCamera, format_report(), GraspMissArm (+15 more)

### Community 26 - "Config Components"
Cohesion: 0.08
Nodes (20): Config, deep_merge(), _find_root(), load_config(), package_root(), _parse_value(), read_yaml(), _set_dotted() (+12 more)

### Community 27 - "Camera Configuration Geometry"
Cohesion: 0.07
Nodes (21): _as_matrix(), camera_pose_in_base(), _check_rigid(), Extrinsics, extrinsics_from_dict(), is_optical_frame(), _matrix_list(), _parse_transform() (+13 more)

### Community 28 - "Extrinsics Components"
Cohesion: 0.08
Nodes (23): camera_pose_in_base(), _check_radians(), Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml(), _link_to_optical() (+15 more)

### Community 29 - "Teach Poses Components"
Cohesion: 0.06
Nodes (5): fmt_mm(), _floats_or(), _Teacher, zone_changes(), zone_name_problem()

### Community 30 - "Grasp Components"
Cohesion: 0.06
Nodes (15): expected_table_z(), _finite(), _float(), GraspPlanner, _identity_plan(), _pad_taper_deg(), release_slots(), _required_float() (+7 more)

### Community 32 - "Scenarios Components"
Cohesion: 0.06
Nodes (27): g(), _arc_capsules(), _banana_parts(), default_intrinsics(), _draw_labels(), _eye_in_hand_observe(), eye_in_hand_T_flange_camera(), _fk_frames() (+19 more)

### Community 33 - "Run Task Components"
Cohesion: 0.09
Nodes (28): RealClock, _arm_text(), _attach_mujoco(), _base_summary(), build_parser(), _classifier_text(), _close_log(), _close_mujoco() (+20 more)

### Community 34 - "Selfcheck Components"
Cohesion: 0.08
Nodes (28): arm_self_check(), at_observe_pose(), _camera_frame(), _check_arm(), _check_assets(), _check_camera(), _check_extrinsics(), check_pick_area() (+20 more)

### Community 35 - "Web Service Routes"
Cohesion: 0.09
Nodes (9): idle_batch(), BusyError, DashboardController, editable_session(), run(), work(), run(), run() (+1 more)

### Community 36 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 37 - "Check Setup Components"
Cohesion: 0.07
Nodes (22): seamless_teach_switch(), teach_exit_hint(), camera_pose_for_check(), _pick_area_depth(), pick_area_mask(), _arm_fix(), build_parser(), camera_topics() (+14 more)

### Community 38 - "Camera Handeye Calibration"
Cohesion: 0.08
Nodes (15): cmd_board(), cmd_handeye(), _collect_live(), board_corners_3d(), BoardSpec, coverage_warning(), detect_board(), _dictionary() (+7 more)

### Community 40 - "Placement Region Observation"
Cohesion: 0.09
Nodes (24): candidate_visibility_filter(), allowed(), _dimensions(), _inspection_geometry(), _inspection_visibility_score(), iter_safe_inspection_poses(), visibility_score(), _planning_call() (+16 more)

### Community 41 - "Make Dataset Components"
Cohesion: 0.09
Nodes (17): build_dataset(), build_parser(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes (+9 more)

### Community 42 - "Validated Skill Execution"
Cohesion: 0.12
Nodes (8): unique_target(), _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search()

### Community 43 - "Logger Components"
Cohesion: 0.08
Nodes (5): _EventLogger, dumps_compact(), RunLogger, safe_filename(), _short()

### Community 44 - "Record Components"
Cohesion: 0.09
Nodes (14): CameraConfig, cmd_record(), _image_to_numpy(), RosRgbd, _atomic_imwrite(), _atomic_text(), depth_to_mm(), next_index() (+6 more)

### Community 45 - "Stack Scene Observation"
Cohesion: 0.08
Nodes (14): _observed_levels(), refine_supported_objects(), _support_evidence(), InstanceClassifier, TablePlane, parse_zones(), _check_ranges(), ColorClassifier (+6 more)

### Community 46 - "Pipeline Components"
Cohesion: 0.12
Nodes (10): _training_samples(), capture_averaged(), _compatible(), Frame, PerceptionResult, _finite(), mask_in_box_overlap(), _ms() (+2 more)

### Community 47 - "Live Components"
Cohesion: 0.08
Nodes (17): apply_live(), drop_live_seen(), drop_unconfirmed_live(), LiveSettings, LiveWatcher, _rot_deg(), _xy(), clearances() (+9 more)

### Community 48 - "Export Onnx Components"
Cohesion: 0.10
Nodes (15): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+7 more)

### Community 49 - "Pyagxarm Arm Components"
Cohesion: 0.09
Nodes (3): _import_sdk(), SdkCallError, _SdkPort

### Community 50 - "Coordinate Transform Utilities"
Cohesion: 0.10
Nodes (20): closing_angle_of(), fit_rigid(), make_T(), pose6_to_T(), quat_to_rot(), rot_to_quat(), rot_to_rpy(), rot_x() (+12 more)

### Community 51 - "Render Components"
Cohesion: 0.11
Nodes (19): _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box(), _hit_capsule(), _hit_cylinder() (+11 more)

### Community 52 - "Classify Knn Components"
Cohesion: 0.10
Nodes (4): KnnClassifier, nan_free(), train_knn_from_sim(), _knn_loadable()

### Community 53 - "Grasp Components"
Cohesion: 0.11
Nodes (11): choose_closing_angle(), neutral_closing_angle(), capsule_edge_margin(), footprint_capsule(), _occupied_capsule(), PlacePlanner, evaluate(), reserve_ok() (+3 more)

### Community 54 - "Motion Ctl Components"
Cohesion: 0.14
Nodes (20): max_gripper_width(), _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick(), cmd_status() (+12 more)

### Community 55 - "Features Components"
Cohesion: 0.09
Nodes (13): _circular_mean_hue(), hsv_ranges_from_samples(), in_ranges(), bgr_to_hsv_pixels(), _crop_mask(), feature_vector(), footprint_shape(), hue_histogram() (+5 more)

### Community 56 - "Sim Camera Components"
Cohesion: 0.08
Nodes (5): make_camera(), _make_sim_camera(), _pose(), CameraSource, SimCamera

### Community 58 - "Simulation World State"
Cohesion: 0.09
Nodes (8): _box_surface(), _capsule_surface(), _cylinder_surface(), densify_polyline(), _dome_taper_deg(), _ellipsoid_surface(), fibonacci_sphere(), pad_section()

### Community 59 - "Preview Simulation Session"
Cohesion: 0.12
Nodes (9): measurement_visible(), merge_local_detections(), _xy(), inspection_history(), remember_inspection(), measure(), phase(), progress() (+1 more)

### Community 60 - "Calib Points Components"
Cohesion: 0.09
Nodes (7): _Calib, find_outliers(), iso_now(), pairs_to_json(), parse_command(), residual_table(), coverage_check()

### Community 61 - "Onsite Components"
Cohesion: 0.08
Nodes (10): camera_pose_for_frame(), capture_view(), compose_view(), depth_range_auto(), draw_numbers(), recorded_flange_pose(), depth_colormap(), side_by_side() (+2 more)

### Community 63 - "Views Components"
Cohesion: 0.11
Nodes (13): coverage_report(), drop_near(), _in_image(), merge_view_detections(), rank(), nearest_view(), polygon_points(), separate_height_layers() (+5 more)

### Community 64 - "Calib Points Components"
Cohesion: 0.10
Nodes (6): average_measurements(), _copy_or_none(), draw_measurement(), Measurement, measurement_line(), grab_frames()

### Community 65 - "Factory Components"
Cohesion: 0.12
Nodes (6): Detector2D, classifier_backend(), make_classifier(), make_detector(), make_perception(), L1NewPerception

### Community 66 - "Ui Components"
Cohesion: 0.09
Nodes (4): is_teaching(), _leave_teach_mode(), display_available(), Ui

### Community 68 - "Language Model Planning"
Cohesion: 0.12
Nodes (10): _iter_sse(), _json_object(), LLMError, _model_placement(), number(), pair(), ModelClient, check_cancel() (+2 more)

### Community 69 - "Live Components"
Cohesion: 0.10
Nodes (3): ArmPoseReader, RosTcpPose, _Pose

### Community 70 - "MuJoCo Scene Construction"
Cohesion: 0.16
Nodes (15): _add_environment(), _add_object(), _add_zone(), _camera(), _camera_housing(), _el(), _find_body(), _look_at() (+7 more)

### Community 72 - "Calib Points Components"
Cohesion: 0.12
Nodes (9): _as_list(), compare_extrinsics(), _deg_between(), plane_basis(), sanity_report(), table_in_base(), _table_rows(), _unit() (+1 more)

### Community 74 - "Recovery Components"
Cohesion: 0.12
Nodes (7): _NoRedirect, _open(), _copy_fields(), model_recovery_context(), _number(), _text(), validate_recovery_decision()

### Community 75 - "Teach Poses Components"
Cohesion: 0.10
Nodes (7): _ViewerClock, SimClock, kinematic_arm(), pose_report(), post_write_report(), reach_report(), selfcheck_failures()

### Community 76 - "Calib Points Components"
Cohesion: 0.11
Nodes (9): clean_depth(), detect_markers(), fit_plane_ransac(), _lsq_plane(), marker_corners_base(), measure_target_aruco(), paint_marker(), quad_center() (+1 more)

### Community 77 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 78 - "Train Yolo Components"
Cohesion: 0.16
Nodes (8): _backup(), build_parser(), config_snippet(), dataset_images(), deploy(), _export_module(), main(), parse_extra()

### Community 79 - "Catalog Perception Components"
Cohesion: 0.14
Nodes (8): _cache_path(), _center_round_fruit(), _classifier(), _ColorSeededKnn, completeness(), make_sandbox_perception(), pear_pose(), SandboxPerception

### Community 80 - "Skill Protocol Components"
Cohesion: 0.24
Nodes (16): _capabilities(), _destination_summary(), _failure_summary(), _fields(), _finite(), _goals(), _index(), _known_label() (+8 more)

### Community 81 - "Table Components"
Cohesion: 0.11
Nodes (5): _LayerPerception, fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), TableFit

### Community 82 - "Video Components"
Cohesion: 0.14
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 83 - "MuJoCo Scene Construction"
Cohesion: 0.15
Nodes (8): assets_dir_for(), build_scene(), _cfg_get(), _compile(), _part_area(), _scenario_world(), scene_materials(), _write_xml()

### Community 85 - "Scenarios Components"
Cohesion: 0.10
Nodes (7): class_grasp(), _image_margin(), layout_reach_report(), layout_view_report(), official_graspability(), official_zones(), _release_xys()

### Community 86 - "Moveit Bridge Components"
Cohesion: 0.12
Nodes (3): execution_waypoints(), MoveItBridge, resample_waypoints()

### Community 87 - "MuJoCo Scene Construction"
Cohesion: 0.12
Nodes (9): _cfg_get(), require_mujoco(), _as_list(), camera_table(), _geom_min_z(), intrinsics_for_fovy(), _make_meta(), mujoco_to_optical() (+1 more)

### Community 88 - "Run Components"
Cohesion: 0.15
Nodes (5): _EventTap, expected_zones(), failure_causes(), _object_cause(), _pick_cause()

### Community 89 - "Teach Poses Components"
Cohesion: 0.11
Nodes (6): dotted_keys(), fmt_list(), write_site_yaml(), check_tasks(), file_header(), use_text()

### Community 90 - "Detector Onnx Components"
Cohesion: 0.16
Nodes (7): _check_image(), decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), unletterbox_boxes()

### Community 91 - "Cli Components"
Cohesion: 0.17
Nodes (8): build_parser(), cmd_show(), cmd_transform(), _collect_offline(), _default(), _load_cam(), main(), list_frames()

### Community 92 - "Detector Onnx Components"
Cohesion: 0.14
Nodes (5): _first_line(), OnnxInfo, _onnxruntime_available(), _OpenCVRunner, _OrtRunner

### Community 93 - "Calib Points Components"
Cohesion: 0.13
Nodes (7): aruco_dictionary(), check_args(), check_point_geometry(), line_offset(), marker_image(), sim_spots(), sim_target()

### Community 94 - "Calib Points Components"
Cohesion: 0.15
Nodes (6): _as_array(), pair_arrays(), pairs_from_json(), PointPair, prediction_error(), solve_extrinsics()

### Community 95 - "Check Onnx Components"
Cohesion: 0.15
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 96 - "Run Components"
Cohesion: 0.16
Nodes (3): _ObjState, PhysicsMonitor, _r()

### Community 97 - "Camera Configuration Geometry"
Cohesion: 0.15
Nodes (7): _apply(), _atomic_write(), load_camera_config(), points_base_to_camera(), points_camera_to_base(), save_camera_config(), write_extrinsics()

### Community 99 - "Grasp Components"
Cohesion: 0.17
Nodes (4): wrap_pi(), Zone, done(), strip_axis()

### Community 101 - "Catalog Components"
Cohesion: 0.20
Nodes (5): CatalogEditor, make_catalog_object(), _number(), _segment_in_rectangle(), _xy()

### Community 102 - "Detector Onnx Components"
Cohesion: 0.26
Nodes (7): build_label_mapping(), _check_metadata(), _check_names(), DetectorError, _probability(), read_labels(), _resolve_input_size()

### Community 103 - "Visualize Components"
Cohesion: 0.27
Nodes (6): as_zone_list(), _draw_base_polygon(), _draw_detection(), draw_result(), label_color(), _text()

### Community 105 - "Standalone Cube Detection"
Cohesion: 0.18
Nodes (3): _load_l1(), _check_ranges(), L1Params

### Community 106 - "Detector Onnx Components"
Cohesion: 0.27
Nodes (6): _fields(), opencv_version(), read_onnx_info(), _text(), _value_info(), _varint()

### Community 107 - "Observations Components"
Cohesion: 0.26
Nodes (7): _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform()

### Community 108 - "Preview Simulation Session"
Cohesion: 0.22
Nodes (3): confirmed(), physically_settled(), audit()

### Community 110 - "MuJoCo Scene Construction"
Cohesion: 0.18
Nodes (5): fovy_deg(), _init_arm(), _object_records(), zone_floor_z(), ObjectRecord

### Community 111 - "MuJoCo Scene Construction"
Cohesion: 0.18
Nodes (5): _compound_part(), object_mass(), object_material(), object_material_name(), _object_parts()

### Community 113 - "Language Model Planning"
Cohesion: 0.20
Nodes (3): LLMSettings, load_settings(), _read_config()

### Community 114 - "Teach Poses Components"
Cohesion: 0.18
Nodes (5): clip_to_limits(), draw_map(), joint_limit_violations(), _joint_limits(), map_px()

### Community 116 - "Motion Ctl Components"
Cohesion: 0.22
Nodes (5): build_parser(), _common_parent(), main(), motion_tuning_file(), _open()

### Community 119 - "Dialogue Context Management"
Cohesion: 0.32
Nodes (3): conversation_summary(), model_conversation(), _safe_text()

### Community 122 - "Check Setup Components"
Cohesion: 0.32
Nodes (4): can_up_command(), check_can(), official_can_script(), _pkg_root()

### Community 123 - "Run Task Components"
Cohesion: 0.25
Nodes (3): home_target(), parse_targets(), Refused

### Community 124 - "Onsite Components"
Cohesion: 0.25
Nodes (4): sim_arm(), sim_drag_joints(), teach_hint(), _sim_joints()

### Community 125 - "Frames Components"
Cohesion: 0.32
Nodes (3): load_frame(), load_image_pair(), RgbdFrame

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1967 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_scene()` connect `MuJoCo Scene Construction` to `Grasp Components`, `MuJoCo Scene Construction`, `Geometry Components`, `Simulation World State`, `MuJoCo Scene Construction`, `Live Components`, `MuJoCo Scene Construction`, `MuJoCo Scene Construction`, `Simulation World State`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Teach Classes Components` be split into smaller, more focused modules?**
  _Cohesion score 0.033185840707964605 - nodes in this community are weakly interconnected._
- **Why does `SimObject` connect `Scenarios Components` to `Coordinate Transform Utilities`, `Simulation World State`, `Simulation World State`, `MuJoCo Scene Construction`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `Obstacle Recovery Components` be split into smaller, more focused modules?**
  _Cohesion score 0.040498442367601244 - nodes in this community are weakly interconnected._
- **Why does `make_arm()` connect `Kinematics Components` to `Sim Arm Components`, `MuJoCo Arm Control`, `Safety Components`, `Pyagxarm Arm Components`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `Onsite Components` be split into smaller, more focused modules?**
  _Cohesion score 0.03997715591090805 - nodes in this community are weakly interconnected._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1347 raw AST edges have unresolved endpoints; 6 self-loops; 756 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
