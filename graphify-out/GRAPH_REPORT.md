# Graph Report - 智能车项目  (2026-11-21)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4852 nodes · 13547 edges · 148 communities (105 shown, 43 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 717 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Color Calib Components
- Teach Poses Components
- Teach Classes Components
- Scenarios Components
- Calib Verify Components
- Grasp Components
- Config Components
- Interfaces Components
- Simulation World State
- MuJoCo Arm Control
- Onsite Components
- Browser Task Controls
- Pyagxarm Arm Components
- Simulation World State
- Segment Components
- Preview Simulation Session
- Legacy Motion Skills
- Teach Classes Components
- Motion Ctl Components
- Safety Components
- Coordinate Transform Utilities
- Legacy Task Runner
- Export Onnx Components
- Teach Record Components
- Standalone Cube Detection
- Sim Bench Components
- Geometry Components
- Extrinsics Components
- Camera Configuration Geometry
- Capture Components
- Pipeline Components
- Simulation World State
- Preview Simulation Session
- Obstacle Recovery Components
- Web Service Routes
- Ik Components
- Ros Camera Components
- Sim Arm Components
- Language Model Planning
- Make Dataset Components
- Placement Region Observation
- Destinations Components
- Kinematics Components
- Run Task Components
- Validated Skill Execution
- Selfcheck Components
- Cli Components
- Legacy Task Runner
- Preview Simulation Session
- Camera Handeye Calibration
- Common Components
- Composable Skill Orchestration
- Legacy Task Runner
- Render Components
- MuJoCo Scene Construction
- Pyagxarm Arm Components
- Record Components
- Classify Knn Components
- Check Setup Components
- Live Components
- Held Recovery Components
- MuJoCo Scene Construction
- Catalog Perception Components
- Detector Onnx Components
- Factory Components
- Ui Components
- Calib Points Components
- Calib Points Components
- Calib Points Components
- MuJoCo Physics Engine
- Features Components
- Views Components
- Coordinate Transform Utilities
- Operator Components
- View Components
- Sequential Batch Planning
- Skill Protocol Components
- Logger Components
- Camera Configuration Geometry
- Table Components
- Run Components
- Selection Components
- Calib Points Components
- Calib Points Components
- Browser Scene Editor
- Recorder Components
- Onsite Components
- Calib Points Components
- Check Onnx Components
- Moveit Bridge Components
- Interactive Physics Session
- Sim Camera Components
- Video Components
- Live Components
- Detector Onnx Components
- Catalog Components
- Logger Components
- Video Components
- Observations Components
- Run Components
- Color Calib Components
- Visualize Components
- Standalone Cube Detection
- Detector Onnx Components
- Classify Color Components
- Calib Points Components
- Check Setup Components
- MuJoCo Scene Construction
- File Camera Components
- MuJoCo Scene Construction
- Calib Points Components
- Check Setup Components
- Language Model Planning
- Selfcheck Components
- Grasp Components
- Pyagxarm Arm Components
- Web Service Routes
- Logger Components
- Recovery Components
- Run Components
- Dialogue Context Management
- Sim Bench Components
- Teach Poses Components
- Utils Components
- Grasp Components
- Safety Components
- Simulation World State
- MuJoCo Arm Control
- View Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 111 edges
2. `TaskRunner` - 81 edges
3. `SimulationSession` - 79 edges
4. `SimObject` - 79 edges
5. `CameraIntrinsics` - 67 edges
6. `MujocoArm` - 62 edges
7. `Frame` - 60 edges
8. `PyAgxArm` - 51 edges
9. `inv_T()` - 50 edges
10. `SafeArm` - 50 edges

## Surprising Connections (you probably didn't know these)
- `_training_samples()` --uses--> `CameraIntrinsics`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/core/types.py
- `make_sandbox_perception()` --uses--> `Perception`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/perception/pipeline.py
- `DashboardPhysics` --uses--> `Config`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/core/config.py
- `DashboardPhysics` --uses--> `SafeArm`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/robot/safety.py
- `DashboardPhysics` --uses--> `SafetyGuard`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/robot/safety.py

## Import Cycles
- None detected.

## Communities (148 total, 43 thin omitted)

### Community 0 - "Color Calib Components"
Cohesion: 0.03
Nodes (41): build_colors_yaml(), calib_header(), circular_mean_hue(), _ColorCalib, Command, compute_ranges(), draw_texts(), effective_classes() (+33 more)

### Community 1 - "Teach Poses Components"
Cohesion: 0.03
Nodes (37): parse_zone(), operator_action(), apply_changes(), ccw_polygon(), clip_to_limits(), distance_to_polygon(), draw_map(), effective() (+29 more)

### Community 2 - "Teach Classes Components"
Cohesion: 0.03
Nodes (39): connect_arm_readonly(), flange_to_tcp(), accepted_labels(), add_sim_ground_truth(), Batch, build_report(), check_label(), clean_label() (+31 more)

### Community 3 - "Scenarios Components"
Cohesion: 0.04
Nodes (54): g(), _arc_capsules(), _arc_points(), _as_list(), _axis_angle(), _banana_parts(), _catalog_object(), _catalog_objects() (+46 more)

### Community 4 - "Calib Verify Components"
Cohesion: 0.04
Nodes (33): approach_z(), corrected_extrinsics(), corrected_transform(), correction_formula(), _deviations(), _fmt_d(), hover_plan(), HoverPlan (+25 more)

### Community 5 - "Grasp Components"
Cohesion: 0.05
Nodes (27): choose_closing_angle(), neutral_closing_angle(), wrap_pi(), Zone, capsule_edge_margin(), _finite(), _float(), footprint_capsule() (+19 more)

### Community 6 - "Config Components"
Cohesion: 0.04
Nodes (32): SimClock, Config, deep_merge(), _find_root(), load_config(), package_root(), _parse_value(), read_yaml() (+24 more)

### Community 7 - "Interfaces Components"
Cohesion: 0.06
Nodes (14): capture_averaged(), _compatible(), ArmInterface, CameraSource, Clock, inv_T(), is_rotation(), pose_error() (+6 more)

### Community 8 - "Simulation World State"
Cohesion: 0.04
Nodes (10): LivePreview, _release_xys(), _check_color(), check_pose(), _finite_array(), _normalize_part(), point_in_polygon(), _positive() (+2 more)

### Community 9 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (5): _cfg_float(), _cfg_vec(), MujocoArm, _Profile, _Trajectory

### Community 10 - "Onsite Components"
Cohesion: 0.04
Nodes (36): observe_tcp_pose(), arm_firmware_version(), ArmPose, average_poses(), backup_file(), backup_path(), detection_rows(), dotted_keys() (+28 more)

### Community 11 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 12 - "Pyagxarm Arm Components"
Cohesion: 0.05
Nodes (10): canonical_pose6(), _import_sdk(), _num(), _opt(), piper_firmware_tuple(), PyAgxArm, _req(), seamless_teach_switch() (+2 more)

### Community 13 - "Simulation World State"
Cohesion: 0.05
Nodes (14): compound_sections(), find_grasp(), find_top_down_grasp(), fully_visible(), _no_grasp_reason(), _box_surface(), _capsule_surface(), _cylinder_surface() (+6 more)

### Community 14 - "Segment Components"
Cohesion: 0.06
Nodes (36): _support_evidence(), _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges() (+28 more)

### Community 15 - "Preview Simulation Session"
Cohesion: 0.05
Nodes (3): footprint_corners(), rank_views(), SimulationSession

### Community 16 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (14): rotation_angle(), GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), _force_value(), gripper_force_for(), is_fault_motion_error() (+6 more)

### Community 17 - "Teach Classes Components"
Cohesion: 0.05
Nodes (10): format_table(), read_pose(), command_kind(), counts_text(), instance_rows(), layout_unchanged(), parse_label_command(), plan_labeling() (+2 more)

### Community 18 - "Motion Ctl Components"
Cohesion: 0.07
Nodes (31): ArmError, CameraError, arm_self_check(), _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit() (+23 more)

### Community 19 - "Safety Components"
Cohesion: 0.06
Nodes (8): MotionResult, make_arm(), interpolate_pose(), _rejected(), _rot_exp(), SafeArm, SafetyGuard, free_tilt_limit()

### Community 20 - "Coordinate Transform Utilities"
Cohesion: 0.07
Nodes (29): seating_requires_feedback(), source_anchor(), spherical_model(), _acceptable(), fit_spherical_anchor(), _hypothesis(), _quality(), _refine() (+21 more)

### Community 21 - "Legacy Task Runner"
Cohesion: 0.07
Nodes (3): TaskReport, _NullLogger, TaskRunner

### Community 22 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 23 - "Teach Record Components"
Cohesion: 0.06
Nodes (19): run_review(), contact_sheet(), crop_of(), instance_records(), load_instances(), load_labels(), load_samples(), loo_predictions() (+11 more)

### Community 24 - "Standalone Cube Detection"
Cohesion: 0.07
Nodes (14): _import_piper_vision(), Intrinsics, color_mask(), CubeDetection, depth_to_base(), draw_result(), fit_table_plane(), _in_polygon() (+6 more)

### Community 25 - "Sim Bench Components"
Cohesion: 0.07
Nodes (23): RunExtras, aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), FaultCamera, format_report() (+15 more)

### Community 26 - "Geometry Components"
Cohesion: 0.07
Nodes (19): _check_intr(), _has_dist(), register_depth_to_color(), camera_info_to_intrinsics(), CameraIntrinsics, backproject(), check_depth(), check_intrinsics() (+11 more)

### Community 27 - "Extrinsics Components"
Cohesion: 0.09
Nodes (21): _check_radians(), Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml(), _link_to_optical(), load_extrinsics() (+13 more)

### Community 28 - "Camera Configuration Geometry"
Cohesion: 0.08
Nodes (19): _apply(), _as_matrix(), _atomic_write(), camera_pose_in_base(), CameraConfig, _check_rigid(), Extrinsics, extrinsics_from_dict() (+11 more)

### Community 29 - "Capture Components"
Cohesion: 0.06
Nodes (19): build_parser(), build_parser(), auto_due(), build_extra(), build_parser(), _Capture, _connect_arm_readonly(), default_out_dir() (+11 more)

### Community 30 - "Pipeline Components"
Cohesion: 0.10
Nodes (11): InstanceClassifier, Frame, PerceptionResult, ColorClassifier, annotate_neighbors(), _finite(), mask_in_box_overlap(), _ms() (+3 more)

### Community 32 - "Preview Simulation Session"
Cohesion: 0.08
Nodes (10): measurement_visible(), merge_local_detections(), _xy(), _PlacementBudget, measure(), phase(), progress(), stationary() (+2 more)

### Community 33 - "Obstacle Recovery Components"
Cohesion: 0.09
Nodes (27): _angle_error(), _at_origin(), _bind(), _cancelled(), _checked_placement(), pose_filter(), execute_clearance(), perform() (+19 more)

### Community 34 - "Web Service Routes"
Cohesion: 0.09
Nodes (9): idle_batch(), DashboardController, editable_session(), run(), work(), run(), run(), on_delta() (+1 more)

### Community 35 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 36 - "Ros Camera Components"
Cohesion: 0.08
Nodes (7): check_depth_scale(), depth_msg_to_meters(), fit_intrinsics(), image_msg_to_bgr(), _image_view(), RosCamera, _stamp_to_sec()

### Community 38 - "Language Model Planning"
Cohesion: 0.07
Nodes (13): _iter_sse(), _json_object(), LLMError, LLMSettings, load_settings(), _model_placement(), number(), pair() (+5 more)

### Community 39 - "Make Dataset Components"
Cohesion: 0.09
Nodes (16): build_dataset(), build_parser(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), _parse_roi() (+8 more)

### Community 40 - "Placement Region Observation"
Cohesion: 0.09
Nodes (21): _coverage(), inspect_held(), measurements_changed(), candidate_visibility_filter(), allowed(), _dimensions(), _inspection_geometry(), inspection_history() (+13 more)

### Community 41 - "Destinations Components"
Cohesion: 0.07
Nodes (14): whole_footprint_at_goal(), whole_footprint_in_zone(), catalog_entries(), CatalogEditor, configure_sandbox(), _segment_in_rectangle(), completion_outline(), destination_label() (+6 more)

### Community 42 - "Kinematics Components"
Cohesion: 0.10
Nodes (5): _as_joints(), _check_pose(), PiperKinematics, rot_log(), _cfg_float()

### Community 43 - "Run Task Components"
Cohesion: 0.10
Nodes (25): main(), motion_tuning_file(), _arm_text(), _base_summary(), build_parser(), _classifier_text(), _close_log(), _close_mujoco() (+17 more)

### Community 44 - "Validated Skill Execution"
Cohesion: 0.13
Nodes (6): _PlanChanged, SkillExecutor, guard(), interrupted(), progress(), search()

### Community 45 - "Selfcheck Components"
Cohesion: 0.09
Nodes (24): at_observe_pose(), _camera_frame(), _check_arm(), _check_assets(), _check_camera(), _check_extrinsics(), _check_reachable(), _check_robot_config() (+16 more)

### Community 46 - "Cli Components"
Cohesion: 0.09
Nodes (18): build_parser(), cmd_handeye(), cmd_l1(), cmd_record(), cmd_show(), cmd_transform(), _collect_offline(), _default() (+10 more)

### Community 47 - "Legacy Task Runner"
Cohesion: 0.09
Nodes (15): overlaps(), object_footprint(), object_prism(), detection_table_z(), polygons_overlap(), prepare_polygon(), PreparedPolygon, _placement_anchor_center() (+7 more)

### Community 48 - "Preview Simulation Session"
Cohesion: 0.08
Nodes (11): unique_target(), forget_inspection(), confirmed(), physically_settled(), audit(), camera_confirmed(), center_supported(), measured_support() (+3 more)

### Community 49 - "Camera Handeye Calibration"
Cohesion: 0.10
Nodes (13): cmd_board(), _collect_live(), board_corners_3d(), BoardSpec, detect_board(), _dictionary(), draw_detection(), find_charuco_corners() (+5 more)

### Community 50 - "Common Components"
Cohesion: 0.08
Nodes (14): DashboardServer, RealClock, main(), main(), check_args(), main(), main(), load_cfg_for() (+6 more)

### Community 52 - "Legacy Task Runner"
Cohesion: 0.10
Nodes (9): Detection, annotate_grasp_neighbors(), _footprint(), footprint_dims(), _has_xy(), held_shape(), _is_stack_base(), line_obstacle_top() (+1 more)

### Community 53 - "Render Components"
Cohesion: 0.11
Nodes (19): _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box(), _hit_capsule(), _hit_cylinder() (+11 more)

### Community 54 - "MuJoCo Scene Construction"
Cohesion: 0.13
Nodes (20): _add_environment(), _add_object(), _add_zone(), _as_list(), _camera(), _camera_housing(), _el(), _f() (+12 more)

### Community 56 - "Record Components"
Cohesion: 0.11
Nodes (12): ArmPoseReader, _atomic_imwrite(), _atomic_text(), depth_preview(), depth_to_mm(), next_index(), _Pose, preview() (+4 more)

### Community 57 - "Classify Knn Components"
Cohesion: 0.11
Nodes (3): KnnClassifier, train_knn_from_sim(), _knn_loadable()

### Community 58 - "Check Setup Components"
Cohesion: 0.09
Nodes (14): main(), _arm_fix(), build_parser(), camera_topics(), check_config(), check_live(), check_modules(), check_ros() (+6 more)

### Community 59 - "Live Components"
Cohesion: 0.08
Nodes (3): _image_to_numpy(), RosRgbd, RosTcpPose

### Community 60 - "Held Recovery Components"
Cohesion: 0.11
Nodes (17): apply_translation_anchor(), held_anchor(), _action(), _ActionFailed, _center(), _destination(), execute_held_recovery(), record() (+9 more)

### Community 61 - "MuJoCo Scene Construction"
Cohesion: 0.09
Nodes (13): assets_dir_for(), build_scene(), _cfg_get(), _compile(), fovy_deg(), _init_arm(), intrinsics_for_fovy(), _make_meta() (+5 more)

### Community 62 - "Catalog Perception Components"
Cohesion: 0.09
Nodes (12): _cache_path(), _center_round_fruit(), _classifier(), _ColorSeededKnn, completeness(), make_sandbox_perception(), pear_pose(), SandboxPerception (+4 more)

### Community 63 - "Detector Onnx Components"
Cohesion: 0.12
Nodes (11): build_label_mapping(), _check_metadata(), _check_names(), DetectorError, _first_line(), OnnxInfo, _OpenCVRunner, _OrtRunner (+3 more)

### Community 64 - "Factory Components"
Cohesion: 0.11
Nodes (7): Detector2D, classifier_backend(), make_classifier(), make_detector(), make_perception(), task_labels(), L1NewPerception

### Community 65 - "Ui Components"
Cohesion: 0.09
Nodes (3): _leave_teach_mode(), display_available(), Ui

### Community 66 - "Calib Points Components"
Cohesion: 0.10
Nodes (12): aruco_dictionary(), check_args(), clean_depth(), detect_markers(), fit_plane_ransac(), _lsq_plane(), marker_image(), measure_target_aruco() (+4 more)

### Community 67 - "Calib Points Components"
Cohesion: 0.10
Nodes (6): average_measurements(), draw_measurement(), Measurement, measurement_line(), parse_command(), grab_frames()

### Community 68 - "Calib Points Components"
Cohesion: 0.11
Nodes (3): _Calib, iso_now(), coverage_check()

### Community 70 - "Features Components"
Cohesion: 0.13
Nodes (11): nan_free(), bgr_to_hsv_pixels(), _crop_mask(), feature_vector(), footprint_shape(), hue_histogram(), instance_pixels(), _num() (+3 more)

### Community 71 - "Views Components"
Cohesion: 0.11
Nodes (12): coverage_report(), drop_near(), _in_image(), merge_view_detections(), rank(), nearest_view(), polygon_points(), separate_height_layers() (+4 more)

### Community 72 - "Coordinate Transform Utilities"
Cohesion: 0.11
Nodes (9): apply_T(), make_T(), orthonormalize(), pose6_to_T(), quat_to_rot(), rot_to_quat(), rot_to_rpy(), rotation_angle() (+1 more)

### Community 74 - "View Components"
Cohesion: 0.11
Nodes (9): load_extrinsics_or_none(), counts_line(), FpsMeter, main(), polygon_center(), status_lines(), table_status(), timing_line() (+1 more)

### Community 75 - "Sequential Batch Planning"
Cohesion: 0.12
Nodes (10): bind_item(), build_items(), completed_position(), _in_completed_zone(), matching_objects(), validate_decision(), valid_zone(), _xyz_is_finite() (+2 more)

### Community 76 - "Skill Protocol Components"
Cohesion: 0.21
Nodes (18): _number(), validate_destination(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _finite(), _goals() (+10 more)

### Community 79 - "Table Components"
Cohesion: 0.15
Nodes (7): points_in_zones(), fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), _restrict(), TableFit, TableParams

### Community 80 - "Run Components"
Cohesion: 0.13
Nodes (6): _EventTap, expected_zones(), failure_causes(), final_truth(), _object_cause(), _pick_cause()

### Community 81 - "Selection Components"
Cohesion: 0.14
Nodes (8): clearances(), is_blacklisted(), _radius(), select_target(), dist(), iso_key(), tie(), _xy()

### Community 82 - "Calib Points Components"
Cohesion: 0.11
Nodes (7): _as_array(), _as_list(), check_point_geometry(), compare_extrinsics(), find_outliers(), line_offset(), pairs_from_json()

### Community 83 - "Calib Points Components"
Cohesion: 0.11
Nodes (5): _copy_or_none(), marker_corners_base(), paint_marker(), sim_touch_pose(), _SimOperator

### Community 84 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 85 - "Recorder Components"
Cohesion: 0.16
Nodes (8): _check_idx(), depth_to_mm(), _imwrite_atomic(), list_frames(), load_frame(), _name(), next_index(), save_frame()

### Community 86 - "Onsite Components"
Cohesion: 0.11
Nodes (8): camera_pose_for_frame(), compose_view(), depth_range_auto(), sim_drag_joints(), sim_drag_tcp(), _sim_joints(), depth_colormap(), side_by_side()

### Community 87 - "Calib Points Components"
Cohesion: 0.11
Nodes (7): TablePlane, _deg_between(), sanity_report(), table_in_base(), _table_rows(), _unit(), view_geometry()

### Community 88 - "Check Onnx Components"
Cohesion: 0.13
Nodes (7): OnnxDetector, build_parser(), _env(), main(), run(), _synthetic_image(), ModelBoxes

### Community 89 - "Moveit Bridge Components"
Cohesion: 0.12
Nodes (3): execution_waypoints(), MoveItBridge, resample_waypoints()

### Community 91 - "Sim Camera Components"
Cohesion: 0.12
Nodes (3): _make_sim_camera(), _pose(), SimCamera

### Community 92 - "Video Components"
Cohesion: 0.15
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 93 - "Live Components"
Cohesion: 0.14
Nodes (7): apply_live(), drop_live_seen(), drop_unconfirmed_live(), LiveSettings, LiveWatcher, _rot_deg(), _xy()

### Community 94 - "Detector Onnx Components"
Cohesion: 0.14
Nodes (7): _check_image(), decode_output(), letterbox(), nms_per_label(), _shape_mismatch_message(), to_pixel_box(), unletterbox_boxes()

### Community 95 - "Catalog Components"
Cohesion: 0.18
Nodes (8): make_catalog_object(), _number(), _xy(), _class_grasp(), _jitter(), official_cube(), official_model(), _probe()

### Community 98 - "Observations Components"
Cohesion: 0.20
Nodes (9): _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform(), points_in_polygon() (+1 more)

### Community 99 - "Run Components"
Cohesion: 0.16
Nodes (3): _ObjState, PhysicsMonitor, _r()

### Community 101 - "Visualize Components"
Cohesion: 0.23
Nodes (7): as_zone_list(), densify_polyline(), _draw_base_polygon(), _draw_detection(), draw_result(), label_color(), _text()

### Community 102 - "Standalone Cube Detection"
Cohesion: 0.15
Nodes (3): _load_l1(), _check_ranges(), L1Params

### Community 103 - "Detector Onnx Components"
Cohesion: 0.24
Nodes (7): _fields(), _onnxruntime_available(), opencv_version(), read_onnx_info(), _text(), _value_info(), _varint()

### Community 104 - "Classify Color Components"
Cohesion: 0.20
Nodes (4): _check_ranges(), _circular_mean_hue(), hsv_ranges_from_samples(), in_ranges()

### Community 105 - "Calib Points Components"
Cohesion: 0.19
Nodes (6): pair_arrays(), pairs_to_json(), PointPair, prediction_error(), residual_table(), solve_extrinsics()

### Community 107 - "MuJoCo Scene Construction"
Cohesion: 0.17
Nodes (5): _cfg_get(), require_mujoco(), camera_table(), _geom_min_z(), piper_geometry()

### Community 109 - "MuJoCo Scene Construction"
Cohesion: 0.18
Nodes (5): _compound_part(), object_mass(), object_material(), object_material_name(), _object_parts()

### Community 110 - "Calib Points Components"
Cohesion: 0.21
Nodes (5): BlockParams, measure_target_block(), _pick_block(), plane_basis(), project_cam()

### Community 111 - "Check Setup Components"
Cohesion: 0.20
Nodes (6): can_up_command(), check_can(), official_can_script(), parse_ip_link(), _pkg_root(), _version()

### Community 112 - "Language Model Planning"
Cohesion: 0.24
Nodes (3): _CancellableStream, _NoRedirect, _open()

### Community 113 - "Selfcheck Components"
Cohesion: 0.25
Nodes (6): make_camera(), camera_pose_for_check(), _pick_area_depth(), pick_area_mask(), _live_camera(), _pick_area_depth()

### Community 114 - "Grasp Components"
Cohesion: 0.24
Nodes (4): expected_table_z(), max_gripper_width(), _required_float(), observe_views()

### Community 117 - "Logger Components"
Cohesion: 0.33
Nodes (3): dumps_compact(), _short(), to_jsonable()

### Community 118 - "Recovery Components"
Cohesion: 0.39
Nodes (5): _copy_fields(), model_recovery_context(), _number(), _text(), validate_recovery_decision()

### Community 120 - "Dialogue Context Management"
Cohesion: 0.32
Nodes (3): conversation_summary(), model_conversation(), _safe_text()

### Community 127 - "Safety Components"
Cohesion: 0.50
Nodes (3): _need(), _need_float(), _need_range()

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1991 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **43 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_open()` connect `Motion Ctl Components` to `Common Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `TaskRunner` (e.g. with `Detection` and `PerceptionResult`) actually correct?**
  _`TaskRunner` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Color Calib Components` be split into smaller, more focused modules?**
  _Cohesion score 0.025734549138804456 - nodes in this community are weakly interconnected._
- **Why does `check_config()` connect `Check Setup Components` to `Check Setup Components`, `Common Components`, `Config Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Should `Teach Poses Components` be split into smaller, more focused modules?**
  _Cohesion score 0.029201101928374655 - nodes in this community are weakly interconnected._
- **Why does `ensure_sim_knn()` connect `Config Components` to `Factory Components`, `Classify Knn Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1372 raw AST edges have unresolved endpoints; 6 self-loops; 756 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
