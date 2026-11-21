# Graph Report - 智能车项目  (2026-11-21)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4906 nodes · 13728 edges · 137 communities (98 shown, 39 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 724 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Teach Poses Components
- Calib Verify Components
- Color Calib Components
- Teach Classes Components
- Coordinate Transform Utilities
- Grasp Components
- Safety Components
- Browser Task Controls
- MuJoCo Arm Control
- Scenarios Components
- Segment Components
- Preview Simulation Session
- Extrinsics Components
- Legacy Motion Skills
- Legacy Task Runner
- Pyagxarm Arm Components
- Standalone Cube Detection
- Camera Configuration Geometry
- Interfaces Components
- Check Setup Components
- Pipeline Components
- Teach Classes Components
- Export Onnx Components
- Sim Bench Components
- Legacy Task Runner
- Run Components
- Language Model Planning
- Placement Region Observation
- Simulation World State
- Validated Skill Execution
- Observation Policy Components
- Selfcheck Components
- Cli Components
- Simulation World State
- Web Service Routes
- Sim Arm Components
- Teach Record Components
- Make Dataset Components
- Composable Skill Orchestration
- Ik Components
- Simulation World State
- Recorder Components
- Ros Camera Components
- Logger Components
- Common Components
- MuJoCo Scene Construction
- Video Components
- Features Components
- Geometry Components
- Calib Points Components
- Geometry Components
- Pyagxarm Arm Components
- Config Components
- Render Components
- Run Task Components
- Kinematics Components
- Camera Handeye Calibration
- Views Components
- Calib Points Components
- Legacy Task Runner
- Calib Points Components
- Skill Protocol Components
- Interactive Physics Session
- Classify Knn Components
- Detector Onnx Components
- Motion Ctl Components
- Calib Points Components
- MuJoCo Scene Construction
- Record Components
- Held Anchor Components
- MuJoCo Physics Engine
- Simulation World State
- Camera Configuration Geometry
- Scenarios Components
- Reach Components
- Held Recovery Components
- Catalog Components
- Simulation World State
- Operator Components
- Ui Components
- Sequential Batch Planning
- Table Components
- Browser Scene Editor
- Live Components
- Detector Onnx Components
- Calib Points Components
- Live Components
- Catalog Perception Components
- Live Components
- Obstacle Recovery Components
- Interfaces Components
- Run Task Components
- Detector Onnx Components
- Factory Components
- Calib Points Components
- Moveit Bridge Components
- Check Onnx Components
- Observations Components
- Preview Components
- Grasp Components
- MuJoCo Scene Construction
- L1 New Components
- MuJoCo Scene Construction
- Onsite Components
- View Components
- Preview Simulation Session
- Pyagxarm Arm Components
- Web Service Routes
- Recovery Components
- Interfaces Components
- Dialogue Context Management
- Onsite Components
- Motion Ctl Components
- Assistant Ui Components
- Scenarios Components
- Common Components
- Teach Poses Components
- Task Contract Components
- Moveit Bridge Components
- Ui Components
- Carried Observation Components
- MuJoCo Arm Control
- Teach Poses Components
- View Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 111 edges
2. `SimulationSession` - 81 edges
3. `TaskRunner` - 81 edges
4. `SimObject` - 79 edges
5. `CameraIntrinsics` - 67 edges
6. `MujocoArm` - 62 edges
7. `Frame` - 60 edges
8. `inv_T()` - 53 edges
9. `PyAgxArm` - 51 edges
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

## Communities (137 total, 39 thin omitted)

### Community 0 - "Teach Poses Components"
Cohesion: 0.02
Nodes (82): parse_zone(), ArmPose, average_poses(), backup_file(), backup_path(), detection_rows(), dotted_keys(), exit_teach_mode_confirmed() (+74 more)

### Community 1 - "Calib Verify Components"
Cohesion: 0.02
Nodes (64): next_index(), build_parser(), main(), approach_z(), build_parser(), check_args(), corrected_extrinsics(), corrected_transform() (+56 more)

### Community 2 - "Color Calib Components"
Cohesion: 0.02
Nodes (44): build_colors_yaml(), build_parser(), calib_header(), circular_mean_hue(), _ColorCalib, Command, compute_ranges(), draw_texts() (+36 more)

### Community 3 - "Teach Classes Components"
Cohesion: 0.03
Nodes (39): camera_pose_for_frame(), recorded_flange_pose(), accepted_labels(), add_sim_ground_truth(), Batch, build_report(), check_label(), clean_label() (+31 more)

### Community 4 - "Coordinate Transform Utilities"
Cohesion: 0.05
Nodes (45): mask_carried_depth(), _pose(), _solid_box_source(), _source_surface_supported(), Clock, closing_angle_of(), fit_rigid(), inv_T() (+37 more)

### Community 5 - "Grasp Components"
Cohesion: 0.05
Nodes (28): choose_closing_angle(), neutral_closing_angle(), wrap_pi(), Zone, capsule_edge_margin(), _finite(), footprint_capsule(), GraspPlanner (+20 more)

### Community 6 - "Safety Components"
Cohesion: 0.04
Nodes (6): ArmInterface, GripperResult, MotionResult, _rejected(), SafeArm, SafetyGuard

### Community 7 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 8 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (5): _cfg_float(), _cfg_vec(), MujocoArm, _Profile, _Trajectory

### Community 9 - "Scenarios Components"
Cohesion: 0.05
Nodes (42): g(), _arc_points(), _as_list(), _axis_angle(), _catalog_object(), _catalog_objects(), _check_graspable(), default_intrinsics() (+34 more)

### Community 10 - "Segment Components"
Cohesion: 0.06
Nodes (36): _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges(), depth_spikes() (+28 more)

### Community 12 - "Extrinsics Components"
Cohesion: 0.05
Nodes (24): _make_sim_camera(), _pose(), _check_radians(), Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml() (+16 more)

### Community 13 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (14): rotation_angle(), GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), _force_value(), gripper_force_for(), is_fault_motion_error() (+6 more)

### Community 14 - "Legacy Task Runner"
Cohesion: 0.06
Nodes (5): TaskReport, _NullLogger, pick_summary(), TaskRunner, _xy()

### Community 15 - "Pyagxarm Arm Components"
Cohesion: 0.07
Nodes (3): Stamped, PyAgxArm, zero_pose_problem()

### Community 16 - "Standalone Cube Detection"
Cohesion: 0.06
Nodes (16): _import_piper_vision(), Intrinsics, _check_ranges(), color_mask(), CubeDetection, depth_to_base(), draw_result(), fit_table_plane() (+8 more)

### Community 17 - "Camera Configuration Geometry"
Cohesion: 0.06
Nodes (25): _apply(), _as_matrix(), camera_pose_in_base(), _check_rigid(), Extrinsics, extrinsics_from_dict(), is_optical_frame(), _matrix_list() (+17 more)

### Community 18 - "Interfaces Components"
Cohesion: 0.06
Nodes (7): capture_averaged(), _compatible(), CameraSource, to_jsonable(), parse_zones(), nan_free(), to_report()

### Community 19 - "Check Setup Components"
Cohesion: 0.05
Nodes (24): main(), camera_pose_for_check(), pick_area_mask(), _arm_fix(), camera_topics(), can_up_command(), check_can(), check_config() (+16 more)

### Community 20 - "Pipeline Components"
Cohesion: 0.08
Nodes (18): _LayerPerception, _observed_levels(), refine_supported_objects(), _support_evidence(), InstanceClassifier, Frame, PerceptionResult, TablePlane (+10 more)

### Community 21 - "Teach Classes Components"
Cohesion: 0.05
Nodes (8): command_kind(), counts_text(), instance_rows(), layout_unchanged(), parse_label_command(), plan_labeling(), _Teacher, wrap_lines()

### Community 22 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 23 - "Sim Bench Components"
Cohesion: 0.06
Nodes (23): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), FaultCamera, format_report(), GraspMissArm (+15 more)

### Community 24 - "Legacy Task Runner"
Cohesion: 0.06
Nodes (23): overlaps(), center_supported(), object_footprint(), object_prism(), detection_table_z(), held_offset_of(), polygons_overlap(), prepare_polygon() (+15 more)

### Community 25 - "Run Components"
Cohesion: 0.06
Nodes (12): _EventTap, expected_zones(), failure_causes(), final_truth(), match_objects(), _object_cause(), _ObjState, PhysicsMonitor (+4 more)

### Community 26 - "Language Model Planning"
Cohesion: 0.06
Nodes (16): _CancellableStream, _iter_sse(), _json_object(), LLMError, LLMSettings, load_settings(), _model_placement(), number() (+8 more)

### Community 27 - "Placement Region Observation"
Cohesion: 0.08
Nodes (25): inspect_held(), measurements_changed(), candidate_visibility_filter(), allowed(), _dimensions(), _inspection_geometry(), inspection_history(), _inspection_visibility_score() (+17 more)

### Community 28 - "Simulation World State"
Cohesion: 0.07
Nodes (8): _check_color(), check_pose(), _dome_taper_deg(), _finite_array(), _normalize_part(), point_in_polygon(), _positive(), SimWorld

### Community 29 - "Validated Skill Execution"
Cohesion: 0.10
Nodes (8): _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search(), _TargetUnavailable

### Community 30 - "Observation Policy Components"
Cohesion: 0.08
Nodes (21): whole_footprint_at_goal(), catalog_entries(), _cache_path(), _classifier(), configure_sandbox(), make_sandbox_perception(), _training_samples(), measurement_visible() (+13 more)

### Community 31 - "Selfcheck Components"
Cohesion: 0.08
Nodes (29): arm_self_check(), at_observe_pose(), _camera_frame(), _check_arm(), _check_assets(), _check_camera(), _check_extrinsics(), check_pick_area() (+21 more)

### Community 32 - "Cli Components"
Cohesion: 0.07
Nodes (21): load_camera_config(), write_extrinsics(), build_parser(), cmd_board(), cmd_handeye(), cmd_l1(), cmd_record(), cmd_show() (+13 more)

### Community 34 - "Web Service Routes"
Cohesion: 0.09
Nodes (11): idle_batch(), BusyError, _dashboard_event(), DashboardController, editable_session(), run(), work(), run() (+3 more)

### Community 36 - "Teach Record Components"
Cohesion: 0.06
Nodes (16): contact_sheet(), crop_of(), instance_records(), load_instances(), load_labels(), load_samples(), loo_predictions(), match_instances() (+8 more)

### Community 37 - "Make Dataset Components"
Cohesion: 0.08
Nodes (17): build_dataset(), build_parser(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes (+9 more)

### Community 39 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 40 - "Simulation World State"
Cohesion: 0.08
Nodes (5): compound_sections(), find_grasp(), find_top_down_grasp(), _no_grasp_reason(), SimObject

### Community 41 - "Recorder Components"
Cohesion: 0.07
Nodes (15): FileCamera, _check_idx(), depth_to_mm(), _imwrite_atomic(), list_frames(), load_frame(), _name(), save_frame() (+7 more)

### Community 42 - "Ros Camera Components"
Cohesion: 0.09
Nodes (7): camera_info_to_intrinsics(), check_depth_scale(), depth_msg_to_meters(), image_msg_to_bgr(), _image_view(), RosCamera, _stamp_to_sec()

### Community 43 - "Logger Components"
Cohesion: 0.08
Nodes (5): _EventLogger, dumps_compact(), RunLogger, safe_filename(), _short()

### Community 44 - "Common Components"
Cohesion: 0.08
Nodes (15): _ViewerClock, make_camera(), SimClock, make_arm(), collect_samples(), make_scenario(), scenario_config_overlay(), collect_sim_samples() (+7 more)

### Community 45 - "MuJoCo Scene Construction"
Cohesion: 0.08
Nodes (18): _as_list(), assets_dir_for(), build_scene(), _cfg_get(), _compile(), fovy_deg(), _geom_min_z(), _init_arm() (+10 more)

### Community 46 - "Video Components"
Cohesion: 0.08
Nodes (4): _LoggerTap, _NullLogger, _PerceptionTap, VideoRecorder

### Community 47 - "Features Components"
Cohesion: 0.08
Nodes (15): _check_ranges(), _circular_mean_hue(), hsv_ranges_from_samples(), in_ranges(), bgr_to_hsv_pixels(), _crop_mask(), feature_vector(), footprint_shape() (+7 more)

### Community 48 - "Geometry Components"
Cohesion: 0.09
Nodes (14): fit_intrinsics(), CameraIntrinsics, backproject(), check_depth(), check_intrinsics(), _dist_array(), has_distortion(), _intrinsics_key() (+6 more)

### Community 49 - "Calib Points Components"
Cohesion: 0.08
Nodes (15): aruco_dictionary(), BlockParams, check_args(), clean_depth(), detect_markers(), fit_plane_ransac(), _lsq_plane(), marker_corners_base() (+7 more)

### Community 50 - "Geometry Components"
Cohesion: 0.08
Nodes (15): _coverage(), whole_footprint_in_zone(), completion_outline(), footprint_in_destination(), whole_footprint_at_destination(), as_zone_list(), densify_polyline(), points_in_polygon() (+7 more)

### Community 51 - "Pyagxarm Arm Components"
Cohesion: 0.09
Nodes (3): _import_sdk(), SdkCallError, _SdkPort

### Community 52 - "Config Components"
Cohesion: 0.12
Nodes (12): Config, deep_merge(), _find_root(), load_config(), package_root(), _parse_value(), read_yaml(), _set_dotted() (+4 more)

### Community 53 - "Render Components"
Cohesion: 0.11
Nodes (19): _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box(), _hit_capsule(), _hit_cylinder() (+11 more)

### Community 54 - "Run Task Components"
Cohesion: 0.13
Nodes (22): _arm_text(), _base_summary(), _classifier_text(), _close_log(), _close_mujoco(), _confirm(), _deg(), exit_code_for() (+14 more)

### Community 55 - "Kinematics Components"
Cohesion: 0.12
Nodes (3): _as_joints(), _check_pose(), PiperKinematics

### Community 56 - "Camera Handeye Calibration"
Cohesion: 0.10
Nodes (11): _collect_live(), board_corners_3d(), BoardSpec, detect_board(), _dictionary(), draw_detection(), find_charuco_corners(), HandEyeCalibrator (+3 more)

### Community 57 - "Views Components"
Cohesion: 0.09
Nodes (16): _footprint(), line_obstacle_top(), _seg_point_dist(), coverage_report(), drop_near(), _in_image(), merge_view_detections(), rank() (+8 more)

### Community 58 - "Calib Points Components"
Cohesion: 0.09
Nodes (6): _Calib, find_outliers(), iso_now(), pairs_to_json(), residual_table(), coverage_check()

### Community 59 - "Legacy Task Runner"
Cohesion: 0.11
Nodes (9): Detection, clearances(), is_blacklisted(), _radius(), select_target(), dist(), iso_key(), tie() (+1 more)

### Community 60 - "Calib Points Components"
Cohesion: 0.09
Nodes (12): _as_array(), _as_list(), check_point_geometry(), compare_extrinsics(), _deg_between(), line_offset(), pairs_from_json(), sanity_report() (+4 more)

### Community 61 - "Skill Protocol Components"
Cohesion: 0.15
Nodes (22): _number(), validate_destination(), plan_limit(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _final_audit_summary() (+14 more)

### Community 62 - "Interactive Physics Session"
Cohesion: 0.09
Nodes (3): DashboardPhysics, forget_inspection(), physically_settled()

### Community 64 - "Detector Onnx Components"
Cohesion: 0.12
Nodes (11): build_label_mapping(), _check_metadata(), _check_names(), DetectorError, _first_line(), OnnxInfo, _OpenCVRunner, _OrtRunner (+3 more)

### Community 65 - "Motion Ctl Components"
Cohesion: 0.16
Nodes (18): _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick(), cmd_status(), cmd_vision() (+10 more)

### Community 66 - "Calib Points Components"
Cohesion: 0.09
Nodes (7): average_measurements(), _copy_or_none(), draw_measurement(), Measurement, measurement_line(), parse_command(), grab_frames()

### Community 67 - "MuJoCo Scene Construction"
Cohesion: 0.14
Nodes (17): _add_environment(), _add_object(), _add_zone(), _camera(), _camera_housing(), _el(), _f(), _find_body() (+9 more)

### Community 68 - "Record Components"
Cohesion: 0.12
Nodes (10): ArmPoseReader, _atomic_imwrite(), _atomic_text(), depth_to_mm(), next_index(), _Pose, record_bag(), record_frames() (+2 more)

### Community 69 - "Held Anchor Components"
Cohesion: 0.10
Nodes (11): apply_translation_anchor(), held_anchor(), seating_requires_feedback(), source_anchor(), spherical_model(), _acceptable(), fit_spherical_anchor(), _hypothesis() (+3 more)

### Community 71 - "Simulation World State"
Cohesion: 0.11
Nodes (7): _box_surface(), _capsule_surface(), _cylinder_surface(), densify_polyline(), _ellipsoid_surface(), fibonacci_sphere(), pad_section()

### Community 73 - "Scenarios Components"
Cohesion: 0.10
Nodes (13): _arc_capsules(), _banana_parts(), layout_bottom_px(), layout_intrinsics(), layout_reach_report(), layout_view_report(), official_observe(), official_parts() (+5 more)

### Community 74 - "Reach Components"
Cohesion: 0.15
Nodes (11): supported_grasp(), accepts_kwarg(), adjust_plan_poses(), _at_z(), _check(), free_tilt_limit(), _heights(), _planning_cancelled() (+3 more)

### Community 75 - "Held Recovery Components"
Cohesion: 0.15
Nodes (16): _action(), _ActionFailed, _center(), _destination(), execute_held_recovery(), record(), stopped(), _fresh_target() (+8 more)

### Community 76 - "Catalog Components"
Cohesion: 0.10
Nodes (7): CatalogEditor, make_catalog_object(), _number(), _segment_in_rectangle(), _xy(), placement_metadata(), _Cancellation

### Community 79 - "Ui Components"
Cohesion: 0.10
Nodes (3): _leave_teach_mode(), display_available(), Ui

### Community 80 - "Sequential Batch Planning"
Cohesion: 0.13
Nodes (9): bind_item(), build_items(), completed_position(), _in_completed_zone(), matching_objects(), _complete_geometry(), validate_decision(), valid_zone() (+1 more)

### Community 81 - "Table Components"
Cohesion: 0.15
Nodes (7): parse_polygon(), fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), _restrict(), TableFit, TableParams

### Community 82 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 83 - "Live Components"
Cohesion: 0.13
Nodes (7): apply_live(), drop_live_seen(), drop_unconfirmed_live(), LiveSettings, LiveWatcher, _rot_deg(), _xy()

### Community 84 - "Detector Onnx Components"
Cohesion: 0.14
Nodes (7): decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), to_pixel_box(), unletterbox_boxes()

### Community 85 - "Calib Points Components"
Cohesion: 0.12
Nodes (5): marker_image(), sim_spots(), sim_target(), sim_touch_pose(), _SimOperator

### Community 86 - "Live Components"
Cohesion: 0.13
Nodes (5): _atomic_write(), CameraConfig, save_camera_config(), _image_to_numpy(), RosRgbd

### Community 87 - "Catalog Perception Components"
Cohesion: 0.16
Nodes (5): _center_round_fruit(), _ColorSeededKnn, completeness(), pear_pose(), SandboxPerception

### Community 89 - "Obstacle Recovery Components"
Cohesion: 0.20
Nodes (10): _angle_error(), _at_origin(), _bind(), _cancelled(), _checked_placement(), pose_filter(), execute_clearance(), perform() (+2 more)

### Community 90 - "Interfaces Components"
Cohesion: 0.13
Nodes (5): RealClock, ArmError, CameraError, _attach_mujoco(), _open()

### Community 91 - "Run Task Components"
Cohesion: 0.15
Nodes (7): home_target(), jog_target(), _open(), parse_targets(), prepare_options(), Refused, resolve_speed()

### Community 92 - "Detector Onnx Components"
Cohesion: 0.22
Nodes (8): _check_image(), _fields(), _onnxruntime_available(), opencv_version(), read_onnx_info(), _text(), _value_info(), _varint()

### Community 93 - "Factory Components"
Cohesion: 0.24
Nodes (6): Detector2D, classifier_backend(), make_classifier(), make_detector(), make_perception(), task_labels()

### Community 94 - "Calib Points Components"
Cohesion: 0.21
Nodes (5): transform_points(), pair_arrays(), PointPair, prediction_error(), solve_extrinsics()

### Community 96 - "Check Onnx Components"
Cohesion: 0.20
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 97 - "Observations Components"
Cohesion: 0.26
Nodes (7): _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform()

### Community 99 - "Grasp Components"
Cohesion: 0.19
Nodes (5): expected_table_z(), _float(), max_gripper_width(), _required_float(), observe_views()

### Community 100 - "MuJoCo Scene Construction"
Cohesion: 0.18
Nodes (5): _compound_part(), object_mass(), object_material(), object_material_name(), _object_parts()

### Community 102 - "MuJoCo Scene Construction"
Cohesion: 0.20
Nodes (4): _cfg_get(), require_mujoco(), camera_table(), mujoco_to_optical()

### Community 103 - "Onsite Components"
Cohesion: 0.22
Nodes (3): compose_view(), depth_range_auto(), text_panel()

### Community 104 - "View Components"
Cohesion: 0.18
Nodes (5): counts_line(), polygon_center(), status_lines(), table_status(), timing_line()

### Community 108 - "Recovery Components"
Cohesion: 0.39
Nodes (5): _copy_fields(), model_recovery_context(), _number(), _text(), validate_recovery_decision()

### Community 110 - "Dialogue Context Management"
Cohesion: 0.32
Nodes (3): conversation_summary(), model_conversation(), _safe_text()

### Community 111 - "Onsite Components"
Cohesion: 0.25
Nodes (3): teach_exit_hint(), arm_firmware_version(), take_can_control_confirmed()

### Community 112 - "Motion Ctl Components"
Cohesion: 0.29
Nodes (4): build_parser(), _common_parent(), main(), motion_tuning_file()

### Community 114 - "Scenarios Components"
Cohesion: 0.29
Nodes (4): class_grasp(), _class_grasp(), official_graspability(), _probe()

### Community 117 - "Task Contract Components"
Cohesion: 0.47
Nodes (3): _all_default_request(), bind_task_contract(), _inventory_labels()

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 2015 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **39 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_catalog_object()` connect `Catalog Components` to `Simulation World State`, `Scenarios Components`, `Segment Components`, `Coordinate Transform Utilities`?**
  _High betweenness centrality (0.000) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `SimulationSession` (e.g. with `CarriedDetections` and `CatalogEditor`) actually correct?**
  _`SimulationSession` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Teach Poses Components` be split into smaller, more focused modules?**
  _Cohesion score 0.018386577798207307 - nodes in this community are weakly interconnected._
- **Why does `SimObject` connect `Simulation World State` to `MuJoCo Scene Construction`, `Simulation World State`, `Scenarios Components`, `Simulation World State`, `Simulation World State`?**
  _High betweenness centrality (0.000) - this node is a cross-community bridge._
- **Should `Calib Verify Components` be split into smaller, more focused modules?**
  _Cohesion score 0.021216496139602033 - nodes in this community are weakly interconnected._
- **Why does `catalog_entries()` connect `Observation Policy Components` to `Catalog Components`?**
  _High betweenness centrality (0.000) - this node is a cross-community bridge._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1388 raw AST edges have unresolved endpoints; 6 self-loops; 758 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
