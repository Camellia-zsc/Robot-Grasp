# Graph Report - 智能车项目  (2026-11-21)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4840 nodes · 13520 edges · 145 communities (100 shown, 45 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 716 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Color Calib Components
- Teach Classes Components
- Teach Poses Components
- Teach Record Components
- Onsite Components
- Scenarios Components
- MuJoCo Arm Control
- Calib Points Components
- Grasp Components
- Simulation World State
- Views Components
- Legacy Task Runner
- Camera Configuration Geometry
- Browser Task Controls
- Simulation World State
- Preview Simulation Session
- Segment Components
- Pyagxarm Arm Components
- Standalone Cube Detection
- Safety Components
- Legacy Motion Skills
- Safety Components
- Record Components
- Export Onnx Components
- Camera Handeye Calibration
- Sim Bench Components
- Calib Points Components
- Web Service Routes
- Pipeline Components
- Simulation World State
- Sim Arm Components
- Teach Classes Components
- Preview Simulation Session
- Geometry Components
- Calib Verify Components
- Language Model Planning
- Preview Simulation Session
- Ik Components
- Placement Region Observation
- Classify Knn Components
- Teach Poses Components
- Run Components
- Held Recovery Components
- MuJoCo Scene Construction
- Make Dataset Components
- Obstacle Recovery Components
- Validated Skill Execution
- Sim Camera Components
- Render Components
- Run Task Components
- Logger Components
- Selfcheck Components
- Legacy Task Runner
- Pyagxarm Arm Components
- Detector Onnx Components
- Calib Points Components
- Kinematics Components
- Table Components
- Coordinate Transform Utilities
- Composable Skill Orchestration
- Extrinsics Components
- Onsite Components
- Motion Ctl Components
- Ros Camera Components
- Config Components
- MuJoCo Physics Engine
- Catalog Perception Components
- Camera Configuration Geometry
- Video Components
- Operator Components
- Skill Protocol Components
- Interactive Physics Session
- Ros Camera Components
- MuJoCo Scene Construction
- Reach Components
- Ui Components
- Spherical Anchor Components
- Browser Scene Editor
- Detector Onnx Components
- Catalog Perception Components
- Moveit Bridge Components
- Check Setup Components
- Color Calib Components
- Color Calib Components
- Color Calib Components
- Teach Record Components
- Video Components
- Catalog Components
- Observations Components
- Extrinsics Components
- Check Onnx Components
- Clock Components
- Onsite Components
- Motion Ctl Components
- Visualize Components
- Check Setup Components
- Features Components
- MuJoCo Scene Construction
- Color Calib Components
- Check Setup Components
- Language Model Planning
- L1 New Components
- Color Calib Components
- Pyagxarm Arm Components
- Color Calib Components
- Motion Ctl Components
- Live Components
- Web Service Routes
- Recovery Components
- Check Setup Components
- Detector Onnx Components
- Dialogue Context Management
- Grasp Components
- Run Components
- Check Setup Components
- Color Calib Components
- Registration Components
- Common Components
- Calib Verify Components
- Color Calib Components
- MuJoCo Arm Control
- Simulation World State
- Selfcheck Components
- MuJoCo Arm Control
- MuJoCo Scene Construction
- Color Calib Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 111 edges
2. `TaskRunner` - 81 edges
3. `SimObject` - 79 edges
4. `SimulationSession` - 78 edges
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

## Communities (145 total, 45 thin omitted)

### Community 0 - "Color Calib Components"
Cohesion: 0.03
Nodes (60): next_index(), RealClock, build_parser(), main(), build_parser(), auto_due(), build_parser(), _Capture (+52 more)

### Community 1 - "Teach Classes Components"
Cohesion: 0.03
Nodes (39): accepted_labels(), add_sim_ground_truth(), Batch, build_report(), check_label(), clean_label(), config_hints(), confirm_labels() (+31 more)

### Community 2 - "Teach Poses Components"
Cohesion: 0.03
Nodes (50): _parse_value(), read_yaml(), write_yaml(), parse_zone(), kinematic_arm(), apply_changes(), ccw_polygon(), check_tasks() (+42 more)

### Community 3 - "Teach Record Components"
Cohesion: 0.04
Nodes (32): _check_idx(), depth_to_mm(), _imwrite_atomic(), list_frames(), load_frame(), _name(), save_frame(), capture_averaged() (+24 more)

### Community 4 - "Onsite Components"
Cohesion: 0.03
Nodes (43): ArmError, check_args(), corrected_extrinsics(), corrected_transform(), correction_formula(), _deviations(), _fmt_d(), main() (+35 more)

### Community 5 - "Scenarios Components"
Cohesion: 0.04
Nodes (54): g(), _arc_capsules(), _as_list(), _axis_angle(), _banana_parts(), _catalog_object(), _catalog_objects(), _check_graspable() (+46 more)

### Community 6 - "MuJoCo Arm Control"
Cohesion: 0.04
Nodes (6): pose_error(), Stamped, _cfg_float(), _cfg_vec(), MujocoArm, _Trajectory

### Community 7 - "Calib Points Components"
Cohesion: 0.04
Nodes (39): aruco_dictionary(), _as_array(), _as_list(), BlockParams, check_args(), check_point_geometry(), clean_depth(), compare_extrinsics() (+31 more)

### Community 8 - "Grasp Components"
Cohesion: 0.05
Nodes (26): choose_closing_angle(), neutral_closing_angle(), wrap_pi(), Zone, capsule_edge_margin(), _finite(), footprint_capsule(), GraspPlanner (+18 more)

### Community 9 - "Simulation World State"
Cohesion: 0.04
Nodes (11): LivePreview, _release_xys(), _check_color(), check_pose(), _dome_taper_deg(), _finite_array(), _normalize_part(), point_in_polygon() (+3 more)

### Community 10 - "Views Components"
Cohesion: 0.05
Nodes (37): Detection, expected_table_z(), apply_live(), drop_live_seen(), drop_unconfirmed_live(), LiveSettings, LiveWatcher, _rot_deg() (+29 more)

### Community 11 - "Legacy Task Runner"
Cohesion: 0.05
Nodes (5): TaskReport, _NullLogger, pick_summary(), TaskRunner, _xy()

### Community 12 - "Camera Configuration Geometry"
Cohesion: 0.05
Nodes (33): _apply(), _as_matrix(), _atomic_write(), camera_pose_in_base(), CameraConfig, _check_rigid(), Extrinsics, extrinsics_from_dict() (+25 more)

### Community 13 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 14 - "Simulation World State"
Cohesion: 0.05
Nodes (14): object_mass(), _object_parts(), compound_sections(), find_grasp(), find_top_down_grasp(), _no_grasp_reason(), _box_surface(), _capsule_surface() (+6 more)

### Community 15 - "Preview Simulation Session"
Cohesion: 0.05
Nodes (5): SimulationSession, measure(), phase(), progress(), camera_pose_in_base()

### Community 16 - "Segment Components"
Cohesion: 0.06
Nodes (35): _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges(), depth_spikes() (+27 more)

### Community 17 - "Pyagxarm Arm Components"
Cohesion: 0.06
Nodes (5): canonical_pose6(), _num(), piper_firmware_tuple(), PyAgxArm, zero_pose_problem()

### Community 18 - "Standalone Cube Detection"
Cohesion: 0.05
Nodes (18): _import_piper_vision(), Intrinsics, cmd_l1(), _load_l1(), _check_ranges(), color_mask(), CubeDetection, depth_to_base() (+10 more)

### Community 19 - "Safety Components"
Cohesion: 0.07
Nodes (18): held_anchor(), ArmInterface, Clock, inv_T(), is_rotation(), ArmHealth, GripperResult, make_arm() (+10 more)

### Community 20 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (13): rotation_angle(), GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), _force_value(), gripper_force_for(), _Lost (+5 more)

### Community 21 - "Safety Components"
Cohesion: 0.06
Nodes (5): MotionResult, _rejected(), SafeArm, SafetyGuard, tcp_error()

### Community 22 - "Record Components"
Cohesion: 0.05
Nodes (18): cmd_record(), list_frames(), ArmPoseReader, _image_to_numpy(), RosRgbd, _atomic_imwrite(), _atomic_text(), depth_preview() (+10 more)

### Community 23 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 24 - "Camera Handeye Calibration"
Cohesion: 0.07
Nodes (18): build_parser(), cmd_board(), cmd_handeye(), _collect_live(), _collect_offline(), _default(), main(), board_corners_3d() (+10 more)

### Community 25 - "Sim Bench Components"
Cohesion: 0.06
Nodes (23): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), FaultCamera, format_report(), GraspMissArm (+15 more)

### Community 26 - "Calib Points Components"
Cohesion: 0.06
Nodes (8): average_measurements(), _Calib, draw_measurement(), Measurement, measurement_line(), parse_command(), grab_frames(), operator_action()

### Community 27 - "Web Service Routes"
Cohesion: 0.08
Nodes (11): idle_batch(), BusyError, DashboardController, editable_session(), run(), work(), run(), run() (+3 more)

### Community 28 - "Pipeline Components"
Cohesion: 0.09
Nodes (11): Detector2D, InstanceClassifier, Frame, PerceptionResult, ColorClassifier, _finite(), mask_in_box_overlap(), _ms() (+3 more)

### Community 31 - "Teach Classes Components"
Cohesion: 0.06
Nodes (6): command_kind(), layout_unchanged(), parse_label_command(), plan_labeling(), _Teacher, wrap_lines()

### Community 32 - "Preview Simulation Session"
Cohesion: 0.08
Nodes (12): measurement_visible(), merge_local_detections(), unique_target(), _xy(), confirmed(), physically_settled(), annotate(), audit() (+4 more)

### Community 33 - "Geometry Components"
Cohesion: 0.07
Nodes (19): _LayerPerception, _observed_levels(), refine_supported_objects(), _support_evidence(), CameraIntrinsics, backproject(), check_depth(), check_intrinsics() (+11 more)

### Community 34 - "Calib Verify Components"
Cohesion: 0.07
Nodes (10): HoverPlan, measure_help(), parse_measurement(), plan_text(), select_targets(), summary_lines(), _Verifier, fmt_list() (+2 more)

### Community 35 - "Language Model Planning"
Cohesion: 0.07
Nodes (13): _iter_sse(), _json_object(), LLMError, LLMSettings, load_settings(), _model_placement(), number(), pair() (+5 more)

### Community 36 - "Preview Simulation Session"
Cohesion: 0.06
Nodes (13): whole_footprint_in_zone(), catalog_entries(), CatalogEditor, configure_sandbox(), _segment_in_rectangle(), completion_outline(), destination_label(), footprint_corners() (+5 more)

### Community 37 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 38 - "Placement Region Observation"
Cohesion: 0.08
Nodes (22): inspect_held(), measurements_changed(), candidate_visibility_filter(), allowed(), _dimensions(), forget_inspection(), _inspection_geometry(), inspection_history() (+14 more)

### Community 39 - "Classify Knn Components"
Cohesion: 0.08
Nodes (8): KnnClassifier, nan_free(), classifier_backend(), make_classifier(), make_detector(), make_perception(), task_labels(), train_knn_from_sim()

### Community 41 - "Run Components"
Cohesion: 0.08
Nodes (10): _EventTap, expected_zones(), failure_causes(), final_truth(), _object_cause(), _ObjState, PhysicsMonitor, _pick_cause() (+2 more)

### Community 42 - "Held Recovery Components"
Cohesion: 0.09
Nodes (23): bind_item(), build_items(), completed_position(), _in_completed_zone(), matching_objects(), whole_footprint_at_goal(), _action(), _ActionFailed (+15 more)

### Community 43 - "MuJoCo Scene Construction"
Cohesion: 0.11
Nodes (22): _add_environment(), _add_object(), _add_zone(), build_scene(), _camera(), _camera_housing(), _el(), _f() (+14 more)

### Community 44 - "Make Dataset Components"
Cohesion: 0.09
Nodes (17): build_dataset(), build_parser(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes (+9 more)

### Community 45 - "Obstacle Recovery Components"
Cohesion: 0.08
Nodes (21): _angle_error(), _at_origin(), _bind(), _cancelled(), _checked_placement(), pose_filter(), execute_clearance(), perform() (+13 more)

### Community 46 - "Validated Skill Execution"
Cohesion: 0.12
Nodes (7): _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search()

### Community 47 - "Sim Camera Components"
Cohesion: 0.06
Nodes (6): make_camera(), _make_sim_camera(), _pose(), FileCamera, CameraSource, SimCamera

### Community 48 - "Render Components"
Cohesion: 0.11
Nodes (21): rot_z(), _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box(), _hit_capsule() (+13 more)

### Community 49 - "Run Task Components"
Cohesion: 0.11
Nodes (25): _arm_text(), _attach_mujoco(), _base_summary(), _classifier_text(), _close_log(), _confirm(), _deg(), exit_code_for() (+17 more)

### Community 50 - "Logger Components"
Cohesion: 0.08
Nodes (4): _EventLogger, RunLogger, safe_filename(), _short()

### Community 51 - "Selfcheck Components"
Cohesion: 0.10
Nodes (24): arm_self_check(), at_observe_pose(), _camera_frame(), _check_arm(), _check_assets(), _check_camera(), _check_extrinsics(), _check_reachable() (+16 more)

### Community 52 - "Legacy Task Runner"
Cohesion: 0.09
Nodes (16): overlaps(), center_supported(), object_footprint(), object_prism(), polygons_overlap(), prepare_polygon(), PreparedPolygon, footprint_dims() (+8 more)

### Community 53 - "Pyagxarm Arm Components"
Cohesion: 0.09
Nodes (3): _import_sdk(), SdkCallError, _SdkPort

### Community 54 - "Detector Onnx Components"
Cohesion: 0.11
Nodes (14): build_label_mapping(), _check_image(), _check_metadata(), _check_names(), DetectorError, _first_line(), OnnxInfo, _onnxruntime_available() (+6 more)

### Community 55 - "Calib Points Components"
Cohesion: 0.06
Nodes (10): _SimOperator, compose_view(), depth_range_auto(), sim_drag_joints(), sim_drag_tcp(), text_panel(), _sim_joints(), use_text() (+2 more)

### Community 56 - "Kinematics Components"
Cohesion: 0.12
Nodes (4): _as_joints(), _check_pose(), PiperKinematics, tilt_outward()

### Community 57 - "Table Components"
Cohesion: 0.10
Nodes (10): TablePlane, parse_polygon(), points_in_zones(), annotate_neighbors(), fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), _restrict() (+2 more)

### Community 58 - "Coordinate Transform Utilities"
Cohesion: 0.11
Nodes (19): closing_angle_of(), fit_rigid(), make_T(), pose6_to_T(), quat_to_rot(), rot_to_quat(), rot_to_rpy(), rot_x() (+11 more)

### Community 60 - "Extrinsics Components"
Cohesion: 0.12
Nodes (12): Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml(), load_extrinsics(), non_optical_frame_message(), normalize_mount() (+4 more)

### Community 61 - "Onsite Components"
Cohesion: 0.08
Nodes (15): seamless_teach_switch(), teach_exit_hint(), _arm_fix(), _live_arm(), arm_firmware_version(), ensure_motion_ready(), exit_teach_mode_confirmed(), health_of() (+7 more)

### Community 62 - "Motion Ctl Components"
Cohesion: 0.16
Nodes (18): _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick(), cmd_status(), cmd_vision() (+10 more)

### Community 63 - "Ros Camera Components"
Cohesion: 0.11
Nodes (3): camera_info_to_intrinsics(), RosCamera, CameraError

### Community 64 - "Config Components"
Cohesion: 0.14
Nodes (7): Config, deep_merge(), load_config(), _set_dotted(), scene_config(), collect_sim_samples(), summarize_cfg()

### Community 66 - "Catalog Perception Components"
Cohesion: 0.09
Nodes (9): _center_round_fruit(), _ColorSeededKnn, completeness(), pear_pose(), SandboxPerception, _check_ranges(), _circular_mean_hue(), hsv_ranges_from_samples() (+1 more)

### Community 70 - "Skill Protocol Components"
Cohesion: 0.21
Nodes (18): _number(), validate_destination(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _finite(), _goals() (+10 more)

### Community 72 - "Ros Camera Components"
Cohesion: 0.15
Nodes (6): check_depth_scale(), depth_msg_to_meters(), fit_intrinsics(), image_msg_to_bgr(), _image_view(), _stamp_to_sec()

### Community 73 - "MuJoCo Scene Construction"
Cohesion: 0.10
Nodes (11): _as_list(), assets_dir_for(), camera_table(), _cfg_get(), _compile(), _compound_part(), _make_meta(), mujoco_to_optical() (+3 more)

### Community 74 - "Reach Components"
Cohesion: 0.16
Nodes (11): parse_zones(), accepts_kwarg(), adjust_plan_poses(), _at_z(), _check(), free_tilt_limit(), _heights(), _planning_cancelled() (+3 more)

### Community 76 - "Spherical Anchor Components"
Cohesion: 0.13
Nodes (10): apply_translation_anchor(), seating_requires_feedback(), source_anchor(), spherical_model(), _acceptable(), fit_spherical_anchor(), _hypothesis(), _quality() (+2 more)

### Community 77 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 78 - "Detector Onnx Components"
Cohesion: 0.13
Nodes (7): decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), to_pixel_box(), unletterbox_boxes()

### Community 79 - "Catalog Perception Components"
Cohesion: 0.12
Nodes (9): _cache_path(), _classifier(), make_sandbox_perception(), _training_samples(), layout_intrinsics(), look_at(), official_wrist_views(), sim_spots() (+1 more)

### Community 80 - "Moveit Bridge Components"
Cohesion: 0.12
Nodes (3): execution_waypoints(), MoveItBridge, resample_waypoints()

### Community 81 - "Check Setup Components"
Cohesion: 0.15
Nodes (10): camera_pose_for_check(), _pick_area_depth(), pick_area_mask(), build_parser(), camera_topics(), check_ros(), _live_camera(), main() (+2 more)

### Community 82 - "Color Calib Components"
Cohesion: 0.14
Nodes (10): circular_mean_hue(), fmt_ranges(), hue_stats(), inside_fraction(), overlap_fractions(), overridden_classes(), plain_ranges(), ranges_array() (+2 more)

### Community 85 - "Teach Record Components"
Cohesion: 0.15
Nodes (5): run_review(), parse_review_command(), parse_sample_id(), Reviewer, suspects()

### Community 86 - "Video Components"
Cohesion: 0.15
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 87 - "Catalog Components"
Cohesion: 0.18
Nodes (8): make_catalog_object(), _number(), _xy(), _class_grasp(), _jitter(), official_cube(), official_model(), _probe()

### Community 88 - "Observations Components"
Cohesion: 0.18
Nodes (9): _coverage(), _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform() (+1 more)

### Community 89 - "Extrinsics Components"
Cohesion: 0.23
Nodes (9): _check_radians(), _link_to_optical(), _numbers(), _parse_transform(), _quat_rot(), _rotation_any(), _tcp_to_flange(), validate_transform() (+1 more)

### Community 90 - "Check Onnx Components"
Cohesion: 0.15
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 91 - "Clock Components"
Cohesion: 0.15
Nodes (4): _ViewerClock, SimClock, collect_samples(), match_objects()

### Community 92 - "Onsite Components"
Cohesion: 0.14
Nodes (5): build_extra(), ArmPose, average_poses(), read_pose(), read_pose_averaged()

### Community 93 - "Motion Ctl Components"
Cohesion: 0.14
Nodes (8): _find_root(), package_root(), build_parser(), _common_parent(), main(), motion_tuning_file(), _open(), _close_mujoco()

### Community 94 - "Visualize Components"
Cohesion: 0.27
Nodes (6): as_zone_list(), _draw_base_polygon(), _draw_detection(), draw_result(), label_color(), _text()

### Community 96 - "Features Components"
Cohesion: 0.23
Nodes (7): _crop_mask(), feature_vector(), footprint_shape(), hue_histogram(), instance_pixels(), _num(), upper_mask()

### Community 97 - "MuJoCo Scene Construction"
Cohesion: 0.17
Nodes (5): _cfg_get(), require_mujoco(), _geom_min_z(), intrinsics_for_fovy(), piper_geometry()

### Community 98 - "Color Calib Components"
Cohesion: 0.15
Nodes (4): effective_classes(), _num(), preview_rows(), view_rows()

### Community 100 - "Check Setup Components"
Cohesion: 0.20
Nodes (4): main(), check_modules(), parse_ip_link(), _version()

### Community 101 - "Language Model Planning"
Cohesion: 0.24
Nodes (3): _CancellableStream, _NoRedirect, _open()

### Community 103 - "Color Calib Components"
Cohesion: 0.20
Nodes (4): draw_texts(), _eroded_upper_pixels(), hue_distance(), instance_hsv()

### Community 105 - "Color Calib Components"
Cohesion: 0.20
Nodes (4): missing_label_warnings(), overlap_pairs(), overlap_warnings(), view_warnings()

### Community 106 - "Motion Ctl Components"
Cohesion: 0.20
Nodes (4): home_target(), jog_target(), parse_targets(), Refused

### Community 109 - "Recovery Components"
Cohesion: 0.39
Nodes (5): _copy_fields(), model_recovery_context(), _number(), _text(), validate_recovery_decision()

### Community 110 - "Check Setup Components"
Cohesion: 0.22
Nodes (4): check_config(), config_fix(), config_messages(), kinematic_arm()

### Community 111 - "Detector Onnx Components"
Cohesion: 0.43
Nodes (5): _fields(), read_onnx_info(), _text(), _value_info(), _varint()

### Community 112 - "Dialogue Context Management"
Cohesion: 0.32
Nodes (3): conversation_summary(), model_conversation(), _safe_text()

### Community 113 - "Grasp Components"
Cohesion: 0.32
Nodes (3): _float(), max_gripper_width(), _required_float()

### Community 115 - "Check Setup Components"
Cohesion: 0.32
Nodes (4): can_up_command(), check_can(), official_can_script(), _pkg_root()

### Community 117 - "Registration Components"
Cohesion: 0.38
Nodes (3): _check_intr(), _has_dist(), register_depth_to_color()

### Community 120 - "Calib Verify Components"
Cohesion: 0.33
Nodes (3): approach_z(), hover_plan(), point_plan()

### Community 125 - "Selfcheck Components"
Cohesion: 0.40
Nodes (3): _point_in_polygon(), _polygons_overlap(), _segments_cross()

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1986 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `collect_samples()` connect `Clock Components` to `Features Components`, `Config Components`, `Camera Configuration Geometry`, `Scenarios Components`, `Run Components`, `Render Components`, `Pipeline Components`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `TaskRunner` (e.g. with `Detection` and `PerceptionResult`) actually correct?**
  _`TaskRunner` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Color Calib Components` be split into smaller, more focused modules?**
  _Cohesion score 0.025062034739454096 - nodes in this community are weakly interconnected._
- **Why does `make_scenario()` connect `Scenarios Components` to `Simulation World State`, `Views Components`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `Teach Classes Components` be split into smaller, more focused modules?**
  _Cohesion score 0.03341523341523341 - nodes in this community are weakly interconnected._
- **Why does `SimObject` connect `Simulation World State` to `Render Components`, `Simulation World State`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1369 raw AST edges have unresolved endpoints; 6 self-loops; 756 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
