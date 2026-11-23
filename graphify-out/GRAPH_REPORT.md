# Graph Report - 智能车项目  (2026-11-23)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4995 nodes · 14058 edges · 155 communities (113 shown, 42 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 760 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Onsite Components
- Onsite Components
- Teach Classes Components
- Grasp Components
- Legacy Task Runner
- Calib Verify Components
- Color Calib Components
- MuJoCo Arm Control
- Browser Task Controls
- Scenarios Components
- Preview Simulation Session
- Safety Components
- Pyagxarm Arm Components
- Segment Components
- Preview Simulation Session
- Safety Components
- Legacy Motion Skills
- Legacy Task Runner
- Check Setup Components
- Color Calib Components
- Teach Classes Components
- Composable Skill Orchestration
- Placement Region Observation
- Geometry Components
- Export Onnx Components
- Pipeline Components
- Selfcheck Components
- Sim Arm Components
- Validated Skill Execution
- Teach Poses Components
- Render Components
- Simulation World State
- Camera Handeye Calibration
- Simulation World State
- Standalone Cube Detection
- Simulation World State
- Web Service Routes
- Views Components
- Ik Components
- Motion Observation Components
- Logger Components
- Teach Poses Components
- Coordinate Transform Utilities
- Extrinsics Components
- Make Dataset Components
- Teach Record Components
- Ros Camera Components
- Sim Bench Components
- Kinematics Components
- Assistant Bench Components
- Features Components
- Pyagxarm Arm Components
- Calib Points Components
- Run Task Components
- Record Components
- Coordinate Transform Utilities
- Interactive Physics Session
- MuJoCo Scene Construction
- Simulation World State
- Simulation World State
- Run Components
- Classify Knn Components
- Camera Configuration Geometry
- Held Recovery Components
- Motion Ctl Components
- Reach Components
- Skill Protocol Components
- Table Components
- Calib Points Components
- Live Components
- MuJoCo Scene Construction
- Detector Onnx Components
- Common Components
- Calib Points Components
- MuJoCo Physics Engine
- Cli Components
- Interfaces Components
- Language Model Planning
- Camera Configuration Geometry
- Catalog Perception Components
- Operator Components
- Recorder Components
- Language Model Planning
- Scenarios Components
- Ui Components
- Browser Scene Editor
- Sequential Batch Planning
- Detector Onnx Components
- Video Components
- Live Components
- File Camera Components
- Moveit Bridge Components
- Config Components
- Video Components
- Sim Camera Components
- Calib Points Components
- Detector Onnx Components
- Observations Components
- Stack Scene Observation
- MuJoCo Scene Construction
- Run Components
- Calib Points Components
- Clock Components
- Calib Points Components
- Onsite Components
- Obstacle Recovery Components
- Check Onnx Components
- Standalone Cube Detection
- Spherical Anchor Components
- Teach Record Components
- Calib Points Components
- Teach Poses Components
- L1 New Components
- Scenarios Components
- Run Task Components
- Grasp Components
- Pyagxarm Arm Components
- MuJoCo Scene Construction
- Motion Ctl Components
- Teach Poses Components
- Web Service Routes
- Recovery Components
- Extrinsics Components
- Factory Components
- Calib Points Components
- Run Components
- Calib Points Components
- Run Task Components
- Sim Bench Components
- Sim Bench Components
- Assistant Ui Components
- Language Model Planning
- Common Components
- Teach Poses Components
- Dialogue Context Management
- Simulation World State
- Ros Camera Components
- MuJoCo Arm Control
- Clock Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 117 edges
2. `SimulationSession` - 90 edges
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

## Communities (155 total, 42 thin omitted)

### Community 0 - "Onsite Components"
Cohesion: 0.03
Nodes (46): next_index(), ArmError, build_parser(), main(), build_parser(), main(), auto_due(), build_extra() (+38 more)

### Community 1 - "Onsite Components"
Cohesion: 0.03
Nodes (48): teach_exit_hint(), arm_firmware_version(), ArmPose, average_poses(), backup_file(), backup_path(), dotted_keys(), ensure_motion_ready() (+40 more)

### Community 2 - "Teach Classes Components"
Cohesion: 0.03
Nodes (39): list_frames(), task_labels(), accepted_labels(), add_sim_ground_truth(), Batch, build_report(), check_label(), clean_label() (+31 more)

### Community 3 - "Grasp Components"
Cohesion: 0.04
Nodes (29): choose_closing_angle(), neutral_closing_angle(), top_down_T(), wrap_pi(), Zone, capsule_edge_margin(), _finite(), footprint_capsule() (+21 more)

### Community 4 - "Legacy Task Runner"
Cohesion: 0.05
Nodes (6): TaskReport, _NullLogger, pick_summary(), TaskRunner, interrupted(), _xy()

### Community 5 - "Calib Verify Components"
Cohesion: 0.04
Nodes (30): approach_z(), check_args(), corrected_extrinsics(), corrected_transform(), correction_formula(), _deviations(), _fmt_d(), hover_plan() (+22 more)

### Community 6 - "Color Calib Components"
Cohesion: 0.04
Nodes (31): build_colors_yaml(), calib_header(), circular_mean_hue(), Command, compute_ranges(), _FlowList, fmt_ranges(), hue_stats() (+23 more)

### Community 7 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (6): MotionResult, _cfg_float(), _cfg_vec(), MujocoArm, _Profile, _Trajectory

### Community 8 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 9 - "Scenarios Components"
Cohesion: 0.05
Nodes (43): _arc_points(), _as_list(), _axis_angle(), _catalog_object(), _catalog_objects(), _check_graspable(), _edge_normals(), _eye_in_hand_observe() (+35 more)

### Community 10 - "Preview Simulation Session"
Cohesion: 0.05
Nodes (3): merge_local_detections(), SimulationSession, via()

### Community 11 - "Safety Components"
Cohesion: 0.06
Nodes (9): ArmInterface, Clock, ArmHealth, GripperResult, Stamped, flange_to_tcp_from_cfg(), _need(), _need_float() (+1 more)

### Community 12 - "Pyagxarm Arm Components"
Cohesion: 0.06
Nodes (6): canonical_pose6(), _num(), _opt(), PyAgxArm, _req(), zero_pose_problem()

### Community 13 - "Segment Components"
Cohesion: 0.06
Nodes (35): _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges(), depth_spikes() (+27 more)

### Community 14 - "Preview Simulation Session"
Cohesion: 0.06
Nodes (20): whole_footprint_at_goal(), catalog_entries(), configure_sandbox(), completion_outline(), destination_label(), footprint_corners(), footprint_in_destination(), placement_metadata() (+12 more)

### Community 15 - "Safety Components"
Cohesion: 0.05
Nodes (6): make_arm(), interpolate_pose(), _rejected(), _rot_exp(), SafeArm, SafetyGuard

### Community 16 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (13): rotation_angle(), GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), _force_value(), gripper_force_for(), _Lost (+5 more)

### Community 17 - "Legacy Task Runner"
Cohesion: 0.06
Nodes (24): _checked_placement(), center_supported(), object_footprint(), object_prism(), detection_table_z(), held_offset_of(), measured_tapered_contact(), transit_height() (+16 more)

### Community 18 - "Check Setup Components"
Cohesion: 0.05
Nodes (25): main(), camera_pose_for_check(), pick_area_mask(), _arm_fix(), build_parser(), camera_topics(), can_up_command(), check_can() (+17 more)

### Community 19 - "Color Calib Components"
Cohesion: 0.05
Nodes (11): _ColorCalib, draw_texts(), effective_classes(), hue_distance(), overlap_pairs(), overlap_warnings(), same_layout(), _View (+3 more)

### Community 20 - "Teach Classes Components"
Cohesion: 0.04
Nodes (11): format_table(), ground_truth_label(), sim_relayout(), command_kind(), counts_text(), instance_rows(), layout_unchanged(), parse_label_command() (+3 more)

### Community 21 - "Composable Skill Orchestration"
Cohesion: 0.08
Nodes (5): _complete_geometry(), goal_worklist(), physically_settled(), plan_limit(), SkillOrchestrator

### Community 22 - "Placement Region Observation"
Cohesion: 0.06
Nodes (30): inspect_held(), measured_neighbors(), measurements_changed(), candidate_visibility_filter(), allowed(), _dimensions(), _grip_bend(), _inspection_geometry() (+22 more)

### Community 23 - "Geometry Components"
Cohesion: 0.07
Nodes (25): _check_intr(), _has_dist(), register_depth_to_color(), camera_info_to_intrinsics(), CameraIntrinsics, as_zone_list(), backproject(), check_depth() (+17 more)

### Community 24 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 25 - "Pipeline Components"
Cohesion: 0.09
Nodes (13): Detector2D, InstanceClassifier, Frame, PerceptionResult, ColorClassifier, make_classifier(), annotate_neighbors(), _finite() (+5 more)

### Community 26 - "Selfcheck Components"
Cohesion: 0.07
Nodes (31): parse_zones(), free_tilt_limit(), arm_self_check(), at_observe_pose(), _camera_frame(), _check_arm(), _check_assets(), _check_camera() (+23 more)

### Community 27 - "Sim Arm Components"
Cohesion: 0.07
Nodes (3): _cfg_float(), _Fault, SimArm

### Community 28 - "Validated Skill Execution"
Cohesion: 0.10
Nodes (8): _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search(), _TargetUnavailable

### Community 29 - "Teach Poses Components"
Cohesion: 0.06
Nodes (25): parse_zone(), ccw_polygon(), distance_to_polygon(), effective(), _finite_number(), get_dotted(), is_convex_polygon(), joint_limit_violations() (+17 more)

### Community 30 - "Render Components"
Cohesion: 0.08
Nodes (26): measurement_absent(), measurement_visible(), rank_views(), _same_layer(), _xy(), _box_prim(), camera_rays(), _check_intrinsics() (+18 more)

### Community 31 - "Simulation World State"
Cohesion: 0.07
Nodes (9): class_grasp(), _class_grasp(), compound_sections(), find_grasp(), find_top_down_grasp(), fully_visible(), _no_grasp_reason(), official_graspability() (+1 more)

### Community 32 - "Camera Handeye Calibration"
Cohesion: 0.07
Nodes (16): Extrinsics, cmd_board(), cmd_handeye(), _collect_live(), board_corners_3d(), BoardSpec, coverage_warning(), detect_board() (+8 more)

### Community 33 - "Simulation World State"
Cohesion: 0.08
Nodes (7): _check_color(), check_pose(), _dome_taper_deg(), _finite_array(), _normalize_part(), _positive(), SimWorld

### Community 34 - "Standalone Cube Detection"
Cohesion: 0.09
Nodes (14): _import_piper_vision(), Intrinsics, color_mask(), CubeDetection, depth_to_base(), draw_result(), fit_table_plane(), _in_polygon() (+6 more)

### Community 36 - "Web Service Routes"
Cohesion: 0.09
Nodes (10): idle_batch(), BusyError, DashboardController, editable_session(), run(), work(), run(), run() (+2 more)

### Community 37 - "Views Components"
Cohesion: 0.07
Nodes (23): Detection, clearances(), is_blacklisted(), _radius(), select_target(), dist(), iso_key(), tie() (+15 more)

### Community 38 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 39 - "Motion Observation Components"
Cohesion: 0.06
Nodes (12): CarriedDetections, _compound_core_supported(), mask_carried_depth(), _pose(), _solid_box_source(), _source_surface_supported(), apply_translation_anchor(), seating_requires_feedback() (+4 more)

### Community 40 - "Logger Components"
Cohesion: 0.08
Nodes (6): _EventLogger, dumps_compact(), RunLogger, safe_filename(), _short(), to_jsonable()

### Community 41 - "Teach Poses Components"
Cohesion: 0.07
Nodes (3): clip_to_limits(), _Teacher, zone_users()

### Community 42 - "Coordinate Transform Utilities"
Cohesion: 0.09
Nodes (19): _as_matrix(), _check_rigid(), extrinsics_from_dict(), is_optical_frame(), _matrix_list(), _parse_transform(), _rotation_of(), _translation_of() (+11 more)

### Community 43 - "Extrinsics Components"
Cohesion: 0.10
Nodes (20): _check_radians(), Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml(), _link_to_optical(), load_extrinsics() (+12 more)

### Community 44 - "Make Dataset Components"
Cohesion: 0.09
Nodes (17): build_dataset(), build_parser(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes (+9 more)

### Community 45 - "Teach Record Components"
Cohesion: 0.08
Nodes (16): contact_sheet(), crop_of(), instance_records(), load_instances(), load_labels(), load_samples(), loo_predictions(), match_instances() (+8 more)

### Community 46 - "Ros Camera Components"
Cohesion: 0.09
Nodes (6): check_depth_scale(), depth_msg_to_meters(), image_msg_to_bgr(), _image_view(), RosCamera, _stamp_to_sec()

### Community 47 - "Sim Bench Components"
Cohesion: 0.09
Nodes (21): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), format_report(), ground_truth(), is_success() (+13 more)

### Community 48 - "Kinematics Components"
Cohesion: 0.09
Nodes (7): _as_joints(), _check_pose(), PiperKinematics, rot_log(), piper_firmware_tuple(), seamless_teach_switch(), _live_arm()

### Community 49 - "Assistant Bench Components"
Cohesion: 0.09
Nodes (18): whole_footprint_in_zone(), CatalogEditor, make_catalog_object(), _number(), restoration_yaw_period(), _segment_in_rectangle(), _xy(), camera_pose_in_base() (+10 more)

### Community 50 - "Features Components"
Cohesion: 0.08
Nodes (14): _check_ranges(), _circular_mean_hue(), hsv_ranges_from_samples(), in_ranges(), bgr_to_hsv_pixels(), _crop_mask(), feature_vector(), footprint_shape() (+6 more)

### Community 51 - "Pyagxarm Arm Components"
Cohesion: 0.09
Nodes (3): _import_sdk(), SdkCallError, _SdkPort

### Community 52 - "Calib Points Components"
Cohesion: 0.09
Nodes (15): aruco_dictionary(), BlockParams, check_args(), clean_depth(), compare_extrinsics(), detect_markers(), fit_plane_ransac(), _lsq_plane() (+7 more)

### Community 53 - "Run Task Components"
Cohesion: 0.13
Nodes (20): _attach_mujoco(), _base_summary(), build_parser(), _close_log(), _close_mujoco(), _confirm(), exit_code_for(), failure_lines() (+12 more)

### Community 54 - "Record Components"
Cohesion: 0.10
Nodes (13): cmd_record(), ArmPoseReader, _atomic_imwrite(), _atomic_text(), depth_to_mm(), next_index(), _Pose, record_bag() (+5 more)

### Community 55 - "Coordinate Transform Utilities"
Cohesion: 0.11
Nodes (17): closing_angle_of(), inv_T(), is_rotation(), make_T(), pose6_to_T(), pose_error(), quat_to_rot(), rot_to_quat() (+9 more)

### Community 56 - "Interactive Physics Session"
Cohesion: 0.08
Nodes (3): DashboardPhysics, _Cancellation, _ViewerClock

### Community 57 - "MuJoCo Scene Construction"
Cohesion: 0.14
Nodes (20): rot_z(), _add_environment(), _add_object(), _add_zone(), _as_list(), _camera(), _camera_housing(), _compound_part() (+12 more)

### Community 58 - "Simulation World State"
Cohesion: 0.08
Nodes (7): _box_surface(), _capsule_surface(), _cylinder_surface(), densify_polyline(), _ellipsoid_surface(), fibonacci_sphere(), pad_section()

### Community 59 - "Simulation World State"
Cohesion: 0.07
Nodes (3): LivePreview, _release_xys(), SimZone

### Community 60 - "Run Components"
Cohesion: 0.10
Nodes (10): collect_samples(), _EventTap, expected_zones(), failure_causes(), final_truth(), match_objects(), _object_cause(), _pick_cause() (+2 more)

### Community 61 - "Classify Knn Components"
Cohesion: 0.12
Nodes (3): KnnClassifier, nan_free(), train_knn_from_sim()

### Community 62 - "Camera Configuration Geometry"
Cohesion: 0.09
Nodes (10): _apply(), _atomic_write(), camera_pose_in_base(), CameraConfig, load_camera_config(), points_base_to_camera(), points_camera_to_base(), save_camera_config() (+2 more)

### Community 63 - "Held Recovery Components"
Cohesion: 0.11
Nodes (19): _action(), _ActionFailed, _center(), _destination(), execute_held_recovery(), record(), stopped(), _fresh_target() (+11 more)

### Community 64 - "Motion Ctl Components"
Cohesion: 0.15
Nodes (18): _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick(), cmd_vision(), _confirm() (+10 more)

### Community 65 - "Reach Components"
Cohesion: 0.12
Nodes (17): gap(), accepts_kwarg(), adjust_plan_poses(), _at_z(), _check(), _heights(), _planning_cancelled(), reach_adjust() (+9 more)

### Community 66 - "Skill Protocol Components"
Cohesion: 0.16
Nodes (22): model_conversation(), _number(), validate_destination(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _final_audit_summary() (+14 more)

### Community 67 - "Table Components"
Cohesion: 0.11
Nodes (8): TablePlane, parse_polygon(), fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), _restrict(), TableFit, TableParams

### Community 69 - "Live Components"
Cohesion: 0.08
Nodes (3): _image_to_numpy(), RosRgbd, RosTcpPose

### Community 70 - "MuJoCo Scene Construction"
Cohesion: 0.10
Nodes (12): assets_dir_for(), build_scene(), _cfg_get(), _compile(), fovy_deg(), _init_arm(), _make_meta(), _part_area() (+4 more)

### Community 71 - "Detector Onnx Components"
Cohesion: 0.13
Nodes (11): build_label_mapping(), _check_metadata(), _check_names(), DetectorError, _first_line(), OnnxInfo, _onnxruntime_available(), _OpenCVRunner (+3 more)

### Community 72 - "Common Components"
Cohesion: 0.12
Nodes (8): Config, _apply_extra(), ensure_sim_knn(), _knn_loadable(), _mujoco_cfg_world(), _scenario_task(), _sim_knn_fingerprint(), summarize_cfg()

### Community 73 - "Calib Points Components"
Cohesion: 0.09
Nodes (7): average_measurements(), _copy_or_none(), draw_measurement(), Measurement, measurement_line(), parse_command(), grab_frames()

### Community 75 - "Cli Components"
Cohesion: 0.13
Nodes (11): build_parser(), cmd_l1(), cmd_show(), cmd_transform(), _collect_offline(), _default(), main(), list_frames() (+3 more)

### Community 76 - "Interfaces Components"
Cohesion: 0.10
Nodes (7): make_camera(), _make_sim_camera(), _pose(), capture_averaged(), _compatible(), median_depth(), CameraSource

### Community 77 - "Language Model Planning"
Cohesion: 0.12
Nodes (10): _iter_sse(), _json_object(), LLMError, _model_placement(), number(), pair(), ModelClient, check_cancel() (+2 more)

### Community 79 - "Catalog Perception Components"
Cohesion: 0.12
Nodes (9): _cache_path(), _center_round_fruit(), _classifier(), _ColorSeededKnn, completeness(), make_sandbox_perception(), pear_pose(), SandboxPerception (+1 more)

### Community 81 - "Recorder Components"
Cohesion: 0.16
Nodes (8): _check_idx(), depth_to_mm(), _imwrite_atomic(), load_frame(), _name(), save_frame(), valid_depth_mask(), atomic_write_text()

### Community 82 - "Language Model Planning"
Cohesion: 0.10
Nodes (5): LLMSettings, load_settings(), _NoRedirect, _open(), _read_config()

### Community 83 - "Scenarios Components"
Cohesion: 0.12
Nodes (12): g(), _arc_capsules(), _banana_parts(), default_intrinsics(), _draw_labels(), layout_intrinsics(), _legacy_scenario(), make_scenario() (+4 more)

### Community 85 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 86 - "Sequential Batch Planning"
Cohesion: 0.14
Nodes (9): bind_item(), build_items(), completed_position(), _in_completed_zone(), matching_objects(), validate_decision(), valid_zone(), _xyz_is_finite() (+1 more)

### Community 87 - "Detector Onnx Components"
Cohesion: 0.14
Nodes (7): decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), to_pixel_box(), unletterbox_boxes()

### Community 88 - "Video Components"
Cohesion: 0.15
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 89 - "Live Components"
Cohesion: 0.14
Nodes (7): apply_live(), drop_live_seen(), drop_unconfirmed_live(), LiveSettings, LiveWatcher, _rot_deg(), _xy()

### Community 91 - "Moveit Bridge Components"
Cohesion: 0.12
Nodes (3): execution_waypoints(), MoveItBridge, resample_waypoints()

### Community 92 - "Config Components"
Cohesion: 0.22
Nodes (8): _find_root(), load_config(), package_root(), _parse_value(), read_yaml(), _set_dotted(), write_yaml(), zone_sources()

### Community 95 - "Calib Points Components"
Cohesion: 0.12
Nodes (4): marker_corners_base(), paint_marker(), sim_touch_pose(), _SimOperator

### Community 96 - "Detector Onnx Components"
Cohesion: 0.18
Nodes (8): _check_image(), _fields(), opencv_version(), _OrtRunner, read_onnx_info(), _text(), _value_info(), _varint()

### Community 97 - "Observations Components"
Cohesion: 0.18
Nodes (9): _coverage(), _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform() (+1 more)

### Community 98 - "Stack Scene Observation"
Cohesion: 0.16
Nodes (7): _box_side_surfaces(), _LayerPerception, _observed_levels(), refine_supported_objects(), _side_support_evidence(), _support_evidence(), transform_cloud()

### Community 99 - "MuJoCo Scene Construction"
Cohesion: 0.12
Nodes (7): _cfg_get(), require_mujoco(), camera_table(), _geom_min_z(), intrinsics_for_fovy(), mujoco_to_optical(), piper_geometry()

### Community 100 - "Run Components"
Cohesion: 0.16
Nodes (3): _ObjState, PhysicsMonitor, _r()

### Community 101 - "Calib Points Components"
Cohesion: 0.15
Nodes (6): _as_array(), _as_list(), check_point_geometry(), line_offset(), pairs_from_json(), pairs_to_json()

### Community 102 - "Clock Components"
Cohesion: 0.16
Nodes (4): SimClock, deep_merge(), scenario_config_overlay(), collect_sim_samples()

### Community 103 - "Calib Points Components"
Cohesion: 0.18
Nodes (6): fit_rigid(), transform_points(), pair_arrays(), PointPair, prediction_error(), solve_extrinsics()

### Community 104 - "Onsite Components"
Cohesion: 0.15
Nodes (5): camera_pose_for_frame(), compose_view(), depth_range_auto(), depth_colormap(), side_by_side()

### Community 105 - "Obstacle Recovery Components"
Cohesion: 0.22
Nodes (9): _angle_error(), _at_origin(), _bind(), _cancelled(), pose_filter(), execute_clearance(), perform(), _holding() (+1 more)

### Community 106 - "Check Onnx Components"
Cohesion: 0.20
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 107 - "Standalone Cube Detection"
Cohesion: 0.18
Nodes (3): _load_l1(), _check_ranges(), L1Params

### Community 108 - "Spherical Anchor Components"
Cohesion: 0.22
Nodes (7): held_anchor(), _acceptable(), fit_spherical_anchor(), _hypothesis(), _quality(), _refine(), validate_anchor_pair()

### Community 110 - "Calib Points Components"
Cohesion: 0.21
Nodes (6): _deg_between(), sanity_report(), table_in_base(), _table_rows(), _unit(), view_geometry()

### Community 111 - "Teach Poses Components"
Cohesion: 0.17
Nodes (5): apply_changes(), check_tasks(), file_header(), pop_dotted(), set_dotted()

### Community 113 - "Scenarios Components"
Cohesion: 0.18
Nodes (6): _image_margin(), layout_reach_report(), entry(), layout_view_report(), official_zones(), _reach_tilt()

### Community 114 - "Run Task Components"
Cohesion: 0.20
Nodes (8): cmd_status(), _arm_text(), _classifier_text(), _deg(), _extrinsics_text(), _observe_text(), _sources(), summary_lines()

### Community 115 - "Grasp Components"
Cohesion: 0.24
Nodes (4): expected_table_z(), _float(), max_gripper_width(), _required_float()

### Community 117 - "MuJoCo Scene Construction"
Cohesion: 0.22
Nodes (4): object_mass(), object_material(), object_material_name(), _object_parts()

### Community 118 - "Motion Ctl Components"
Cohesion: 0.22
Nodes (5): build_parser(), _common_parent(), main(), motion_tuning_file(), _open()

### Community 119 - "Teach Poses Components"
Cohesion: 0.20
Nodes (3): _floats_or(), zone_changes(), zone_name_problem()

### Community 121 - "Recovery Components"
Cohesion: 0.39
Nodes (5): _copy_fields(), model_recovery_context(), _number(), _text(), validate_recovery_decision()

### Community 123 - "Factory Components"
Cohesion: 0.31
Nodes (3): classifier_backend(), make_detector(), make_perception()

### Community 124 - "Calib Points Components"
Cohesion: 0.22
Nodes (4): point_in_polygon(), marker_image(), sim_spots(), sim_target()

### Community 126 - "Calib Points Components"
Cohesion: 0.25
Nodes (3): find_outliers(), iso_now(), residual_table()

### Community 127 - "Run Task Components"
Cohesion: 0.25
Nodes (3): home_target(), parse_targets(), Refused

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 2050 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SimulationSession` connect `Preview Simulation Session` to `Grasp Components`, `Legacy Task Runner`, `Views Components`, `Motion Observation Components`, `Preview Simulation Session`, `Operator Components`, `Assistant Bench Components`, `Legacy Task Runner`, `Composable Skill Orchestration`, `Placement Region Observation`, `Geometry Components`, `Interactive Physics Session`?**
  _High betweenness centrality (0.003) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `SimulationSession` (e.g. with `CarriedDetections` and `CatalogEditor`) actually correct?**
  _`SimulationSession` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Onsite Components` be split into smaller, more focused modules?**
  _Cohesion score 0.03230479501665942 - nodes in this community are weakly interconnected._
- **Why does `build_scene()` connect `MuJoCo Scene Construction` to `Simulation World State`, `MuJoCo Scene Construction`, `Grasp Components`, `Simulation World State`, `Scenarios Components`, `Simulation World State`, `MuJoCo Scene Construction`, `MuJoCo Scene Construction`, `Render Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Should `Onsite Components` be split into smaller, more focused modules?**
  _Cohesion score 0.030727155727155727 - nodes in this community are weakly interconnected._
- **Why does `make_scenario()` connect `Scenarios Components` to `Scenarios Components`, `Simulation World State`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1415 raw AST edges have unresolved endpoints; 6 self-loops; 774 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
