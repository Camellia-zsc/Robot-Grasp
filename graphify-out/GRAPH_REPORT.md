# Graph Report - 智能车项目  (2026-11-21)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4867 nodes · 13594 edges · 146 communities (101 shown, 45 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 719 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Teach Poses Components
- Color Calib Components
- Onsite Components
- Simulation World State
- Scenarios Components
- Safety Components
- Render Components
- Coordinate Transform Utilities
- Grasp Components
- Browser Task Controls
- MuJoCo Arm Control
- Config Components
- Pyagxarm Arm Components
- Segment Components
- Teach Classes Components
- Legacy Motion Skills
- Legacy Task Runner
- Onsite Components
- Calib Verify Components
- Teach Record Components
- Destinations Components
- Preview Simulation Session
- Teach Classes Components
- Extrinsics Components
- Export Onnx Components
- Legacy Task Runner
- Sim Bench Components
- Held Anchor Components
- Camera Configuration Geometry
- Language Model Planning
- Record Components
- Standalone Cube Detection
- Sim Arm Components
- Simulation World State
- Preview Simulation Session
- Web Service Routes
- Ik Components
- Run Task Components
- Camera Handeye Calibration
- Make Dataset Components
- Views Components
- Logger Components
- Validated Skill Execution
- Pipeline Components
- Selfcheck Components
- Kinematics Components
- MuJoCo Scene Construction
- Reach Components
- Classify Color Components
- Detector Onnx Components
- Calib Points Components
- Sim Camera Components
- Pyagxarm Arm Components
- Composable Skill Orchestration
- Classify Knn Components
- Geometry Components
- Run Components
- MuJoCo Scene Construction
- Calib Points Components
- Simulation World State
- Skill Protocol Components
- File Camera Components
- Motion Ctl Components
- Cli Components
- Web Service Routes
- Held Recovery Components
- Calib Points Components
- MuJoCo Physics Engine
- Check Setup Components
- Teach Record Components
- Features Components
- Camera Configuration Geometry
- Calib Points Components
- Calib Points Components
- Placement Region Observation
- Operator Components
- Onsite Components
- Ui Components
- Ros Camera Components
- Ros Camera Components
- Preview Simulation Session
- Browser Scene Editor
- Table Components
- Detector Onnx Components
- Grasp Components
- Video Components
- Preview Simulation Session
- Legacy Task Runner
- Video Components
- Live Components
- Interactive Physics Session
- Check Onnx Components
- Standalone Cube Detection
- Run Components
- Selfcheck Components
- Live Components
- Camera Configuration Geometry
- Stack Scene Observation
- Moveit Bridge Components
- Selection Components
- Check Setup Components
- Factory Components
- MuJoCo Scene Construction
- Interfaces Components
- Visualize Components
- Motion Ctl Components
- Calib Points Components
- Utils Components
- Motion Ctl Components
- Catalog Perception Components
- Calib Points Components
- Pyagxarm Arm Components
- Recovery Components
- L1 New Components
- Pyagxarm Arm Components
- Check Setup Components
- Detector Onnx Components
- Run Components
- Assistant Ui Components
- Teach Poses Components
- Dialogue Context Management
- MuJoCo Arm Control
- Simulation World State
- Selfcheck Components
-   Main   Components
- Carried Observation Components
- MuJoCo Arm Control
- Onsite Components
- Teach Classes Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 111 edges
2. `TaskRunner` - 81 edges
3. `SimulationSession` - 80 edges
4. `SimObject` - 79 edges
5. `CameraIntrinsics` - 67 edges
6. `MujocoArm` - 62 edges
7. `Frame` - 60 edges
8. `inv_T()` - 52 edges
9. `PyAgxArm` - 51 edges
10. `SafeArm` - 50 edges

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

## Communities (146 total, 45 thin omitted)

### Community 0 - "Teach Poses Components"
Cohesion: 0.02
Nodes (56): parse_zone(), dotted_keys(), fmt_list(), fmt_mm(), kinematic_arm(), operator_action(), read_yaml_or_empty(), write_report() (+48 more)

### Community 1 - "Color Calib Components"
Cohesion: 0.02
Nodes (40): build_colors_yaml(), calib_header(), _ColorCalib, Command, compute_ranges(), draw_texts(), effective_classes(), _FlowList (+32 more)

### Community 2 - "Onsite Components"
Cohesion: 0.03
Nodes (54): task_labels(), build_parser(), main(), build_parser(), check_args(), main(), auto_due(), build_extra() (+46 more)

### Community 3 - "Simulation World State"
Cohesion: 0.03
Nodes (16): object_mass(), object_material_name(), _class_grasp(), compound_sections(), find_grasp(), find_top_down_grasp(), _no_grasp_reason(), _box_surface() (+8 more)

### Community 4 - "Scenarios Components"
Cohesion: 0.04
Nodes (58): g(), release_slots(), _arc_capsules(), _as_list(), _axis_angle(), _banana_parts(), _catalog_object(), _catalog_objects() (+50 more)

### Community 5 - "Safety Components"
Cohesion: 0.04
Nodes (6): ArmInterface, ArmHealth, MotionResult, _rejected(), SafeArm, SafetyGuard

### Community 6 - "Render Components"
Cohesion: 0.04
Nodes (22): LivePreview, _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box(), _hit_capsule() (+14 more)

### Community 7 - "Coordinate Transform Utilities"
Cohesion: 0.05
Nodes (41): mask_carried_depth(), _pose(), _solid_box_source(), _source_surface_supported(), closing_angle_of(), fit_rigid(), inv_T(), is_rotation() (+33 more)

### Community 8 - "Grasp Components"
Cohesion: 0.05
Nodes (25): choose_closing_angle(), neutral_closing_angle(), wrap_pi(), Zone, capsule_edge_margin(), _finite(), footprint_capsule(), GraspPlanner (+17 more)

### Community 9 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 10 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (4): _cfg_float(), _cfg_vec(), MujocoArm, _Trajectory

### Community 11 - "Config Components"
Cohesion: 0.05
Nodes (23): _ViewerClock, SimClock, Config, deep_merge(), _find_root(), load_config(), package_root(), _parse_value() (+15 more)

### Community 12 - "Pyagxarm Arm Components"
Cohesion: 0.06
Nodes (6): ArmError, _import_sdk(), _num(), _opt(), PyAgxArm, _req()

### Community 13 - "Segment Components"
Cohesion: 0.06
Nodes (34): _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges(), depth_spikes() (+26 more)

### Community 14 - "Teach Classes Components"
Cohesion: 0.04
Nodes (22): accepted_labels(), build_report(), confused_pairs(), dir_summary(), draw_instances(), evaluate_samples(), evaluation_text(), frame_samples() (+14 more)

### Community 15 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (13): GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), _force_value(), gripper_force_for(), is_fault_motion_error(), _Lost (+5 more)

### Community 17 - "Onsite Components"
Cohesion: 0.05
Nodes (26): arm_firmware_version(), ArmPose, average_poses(), camera_pose_for_frame(), compose_view(), depth_range_auto(), detection_rows(), draw_numbers() (+18 more)

### Community 18 - "Calib Verify Components"
Cohesion: 0.04
Nodes (20): corrected_extrinsics(), correction_formula(), _deviations(), HoverPlan, _mean3(), measure_help(), parent_pose_in_base(), parse_measurement() (+12 more)

### Community 19 - "Teach Record Components"
Cohesion: 0.06
Nodes (24): _check_idx(), depth_to_mm(), _imwrite_atomic(), list_frames(), load_frame(), _name(), next_index(), save_frame() (+16 more)

### Community 20 - "Destinations Components"
Cohesion: 0.06
Nodes (27): _in_completed_zone(), whole_footprint_at_goal(), whole_footprint_in_zone(), catalog_entries(), make_catalog_object(), _number(), _cache_path(), _classifier() (+19 more)

### Community 22 - "Teach Classes Components"
Cohesion: 0.05
Nodes (10): command_kind(), confirm_labels(), counts_text(), instance_rows(), layout_of(), layout_unchanged(), plan_labeling(), rejected_summary() (+2 more)

### Community 23 - "Extrinsics Components"
Cohesion: 0.07
Nodes (25): _check_intr(), _has_dist(), register_depth_to_color(), camera_pose_in_base(), _check_radians(), Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict() (+17 more)

### Community 24 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 25 - "Legacy Task Runner"
Cohesion: 0.07
Nodes (15): Detection, detection_table_z(), held_offset_of(), _footprint(), footprint_dims(), _has_xy(), held_shape(), _is_stack_base() (+7 more)

### Community 26 - "Sim Bench Components"
Cohesion: 0.06
Nodes (23): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), FaultCamera, format_report(), GraspMissArm (+15 more)

### Community 27 - "Held Anchor Components"
Cohesion: 0.05
Nodes (23): apply_translation_anchor(), held_anchor(), seating_requires_feedback(), source_anchor(), spherical_model(), _acceptable(), fit_spherical_anchor(), _hypothesis() (+15 more)

### Community 28 - "Camera Configuration Geometry"
Cohesion: 0.07
Nodes (21): _as_matrix(), camera_pose_in_base(), _check_rigid(), Extrinsics, extrinsics_from_dict(), is_optical_frame(), _matrix_list(), _parse_transform() (+13 more)

### Community 29 - "Language Model Planning"
Cohesion: 0.06
Nodes (16): _CancellableStream, _iter_sse(), _json_object(), LLMError, LLMSettings, load_settings(), _model_placement(), number() (+8 more)

### Community 30 - "Record Components"
Cohesion: 0.07
Nodes (19): CameraConfig, cmd_record(), _collect_live(), ArmPoseReader, _image_to_numpy(), RosRgbd, _atomic_imwrite(), _atomic_text() (+11 more)

### Community 31 - "Standalone Cube Detection"
Cohesion: 0.07
Nodes (13): Intrinsics, color_mask(), CubeDetection, depth_to_base(), draw_result(), fit_table_plane(), _in_polygon(), L1Detector (+5 more)

### Community 32 - "Sim Arm Components"
Cohesion: 0.07
Nodes (3): Stamped, _Fault, SimArm

### Community 34 - "Preview Simulation Session"
Cohesion: 0.07
Nodes (15): _coverage(), inspect_held(), measurements_changed(), measurement_visible(), merge_local_detections(), _xy(), inspection_history(), remember_inspection() (+7 more)

### Community 35 - "Web Service Routes"
Cohesion: 0.10
Nodes (10): idle_batch(), BusyError, DashboardController, editable_session(), run(), work(), run(), run() (+2 more)

### Community 36 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 37 - "Run Task Components"
Cohesion: 0.10
Nodes (26): RealClock, _arm_text(), _attach_mujoco(), _base_summary(), _classifier_text(), _close_log(), _confirm(), _deg() (+18 more)

### Community 38 - "Camera Handeye Calibration"
Cohesion: 0.08
Nodes (14): cmd_board(), cmd_handeye(), board_corners_3d(), BoardSpec, coverage_warning(), detect_board(), _dictionary(), draw_detection() (+6 more)

### Community 39 - "Make Dataset Components"
Cohesion: 0.09
Nodes (17): build_dataset(), build_parser(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes (+9 more)

### Community 40 - "Views Components"
Cohesion: 0.08
Nodes (21): _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform(), _seg_point_dist() (+13 more)

### Community 41 - "Logger Components"
Cohesion: 0.08
Nodes (5): _EventLogger, dumps_compact(), RunLogger, safe_filename(), _short()

### Community 42 - "Validated Skill Execution"
Cohesion: 0.12
Nodes (7): _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search()

### Community 43 - "Pipeline Components"
Cohesion: 0.13
Nodes (10): Frame, PerceptionResult, annotate_neighbors(), _finite(), mask_in_box_overlap(), _ms(), Perception, table_coverage() (+2 more)

### Community 44 - "Selfcheck Components"
Cohesion: 0.10
Nodes (22): parse_zones(), arm_self_check(), _check_arm(), _check_assets(), _check_extrinsics(), _check_reachable(), _check_robot_config(), _check_task_config() (+14 more)

### Community 45 - "Kinematics Components"
Cohesion: 0.11
Nodes (3): _as_joints(), _check_pose(), PiperKinematics

### Community 46 - "MuJoCo Scene Construction"
Cohesion: 0.12
Nodes (21): _add_environment(), _add_object(), _add_zone(), _as_list(), _camera(), _camera_housing(), _compound_part(), _el() (+13 more)

### Community 47 - "Reach Components"
Cohesion: 0.10
Nodes (22): _angle_error(), _at_origin(), _bind(), _cancelled(), _checked_placement(), pose_filter(), execute_clearance(), perform() (+14 more)

### Community 48 - "Classify Color Components"
Cohesion: 0.08
Nodes (9): _ColorSeededKnn, Detector2D, InstanceClassifier, _check_ranges(), _circular_mean_hue(), ColorClassifier, hsv_ranges_from_samples(), in_ranges() (+1 more)

### Community 49 - "Detector Onnx Components"
Cohesion: 0.11
Nodes (14): build_label_mapping(), _check_image(), _check_metadata(), _check_names(), DetectorError, _first_line(), OnnxInfo, _onnxruntime_available() (+6 more)

### Community 50 - "Calib Points Components"
Cohesion: 0.08
Nodes (14): TablePlane, BlockParams, check_args(), clean_depth(), detect_markers(), fit_plane_ransac(), _lsq_plane(), measure_target_aruco() (+6 more)

### Community 51 - "Sim Camera Components"
Cohesion: 0.07
Nodes (5): make_camera(), _make_sim_camera(), _pose(), CameraSource, SimCamera

### Community 54 - "Classify Knn Components"
Cohesion: 0.10
Nodes (4): KnnClassifier, nan_free(), _knn_loadable(), load_existing()

### Community 55 - "Geometry Components"
Cohesion: 0.12
Nodes (16): CameraIntrinsics, as_zone_list(), backproject(), check_depth(), check_intrinsics(), densify_polyline(), _dist_array(), has_distortion() (+8 more)

### Community 56 - "Run Components"
Cohesion: 0.10
Nodes (10): collect_samples(), _EventTap, expected_zones(), failure_causes(), final_truth(), match_objects(), _object_cause(), _pick_cause() (+2 more)

### Community 57 - "MuJoCo Scene Construction"
Cohesion: 0.09
Nodes (14): assets_dir_for(), build_scene(), _cfg_get(), _compile(), fovy_deg(), _init_arm(), intrinsics_for_fovy(), _make_meta() (+6 more)

### Community 58 - "Calib Points Components"
Cohesion: 0.09
Nodes (11): _as_array(), _as_list(), check_point_geometry(), find_outliers(), iso_now(), line_offset(), pairs_from_json(), pairs_to_json() (+3 more)

### Community 59 - "Simulation World State"
Cohesion: 0.10
Nodes (5): _check_color(), _finite_array(), _normalize_part(), _positive(), SimZone

### Community 60 - "Skill Protocol Components"
Cohesion: 0.15
Nodes (20): model_conversation(), validate_destination(), plan_limit(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _finite() (+12 more)

### Community 61 - "File Camera Components"
Cohesion: 0.09
Nodes (3): FileCamera, CameraError, Clock

### Community 62 - "Motion Ctl Components"
Cohesion: 0.16
Nodes (18): _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick(), cmd_status(), cmd_vision() (+10 more)

### Community 63 - "Cli Components"
Cohesion: 0.12
Nodes (12): build_parser(), cmd_l1(), cmd_show(), cmd_transform(), _collect_offline(), _default(), _load_cam(), main() (+4 more)

### Community 64 - "Web Service Routes"
Cohesion: 0.09
Nodes (9): bind_item(), build_items(), completed_position(), matching_objects(), validate_decision(), valid_zone(), _xyz_is_finite(), _dashboard_event() (+1 more)

### Community 65 - "Held Recovery Components"
Cohesion: 0.13
Nodes (18): _action(), _ActionFailed, _center(), _destination(), execute_held_recovery(), record(), stopped(), _fresh_target() (+10 more)

### Community 66 - "Calib Points Components"
Cohesion: 0.10
Nodes (6): average_measurements(), _copy_or_none(), draw_measurement(), Measurement, measurement_line(), grab_frames()

### Community 68 - "Check Setup Components"
Cohesion: 0.12
Nodes (12): build_parser(), camera_topics(), can_up_command(), check_can(), check_modules(), check_ros(), main(), official_can_script() (+4 more)

### Community 69 - "Teach Record Components"
Cohesion: 0.11
Nodes (7): check_label(), run_review(), loo_predictions(), parse_review_command(), parse_sample_id(), Reviewer, suspects()

### Community 70 - "Features Components"
Cohesion: 0.14
Nodes (10): bgr_to_hsv_pixels(), _crop_mask(), feature_vector(), footprint_shape(), hue_histogram(), instance_pixels(), _num(), upper_mask() (+2 more)

### Community 72 - "Calib Points Components"
Cohesion: 0.10
Nodes (3): _Calib, compare_extrinsics(), coverage_check()

### Community 73 - "Calib Points Components"
Cohesion: 0.10
Nodes (6): marker_corners_base(), marker_image(), paint_marker(), sim_touch_pose(), sim_visible(), _SimOperator

### Community 74 - "Placement Region Observation"
Cohesion: 0.15
Nodes (16): candidate_visibility_filter(), allowed(), _dimensions(), forget_inspection(), _inspection_geometry(), _inspection_visibility_score(), iter_safe_inspection_poses(), visibility_score() (+8 more)

### Community 76 - "Onsite Components"
Cohesion: 0.10
Nodes (10): aruco_dictionary(), backup_file(), backup_path(), extrinsics_out_path(), logging_dir(), sim_site_dir(), wall_stamp(), write_extrinsics_site() (+2 more)

### Community 77 - "Ui Components"
Cohesion: 0.10
Nodes (3): _leave_teach_mode(), display_available(), Ui

### Community 78 - "Ros Camera Components"
Cohesion: 0.15
Nodes (6): check_depth_scale(), depth_msg_to_meters(), fit_intrinsics(), image_msg_to_bgr(), _image_view(), _stamp_to_sec()

### Community 80 - "Preview Simulation Session"
Cohesion: 0.12
Nodes (4): unique_target(), confirmed(), physically_settled(), audit()

### Community 81 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 82 - "Table Components"
Cohesion: 0.16
Nodes (7): parse_polygon(), fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), _restrict(), TableFit, TableParams

### Community 83 - "Detector Onnx Components"
Cohesion: 0.13
Nodes (7): decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), to_pixel_box(), unletterbox_boxes()

### Community 84 - "Grasp Components"
Cohesion: 0.13
Nodes (7): expected_table_z(), _float(), max_gripper_width(), _required_float(), free_tilt_limit(), _NullLogger, pick_summary()

### Community 85 - "Video Components"
Cohesion: 0.15
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 86 - "Preview Simulation Session"
Cohesion: 0.11
Nodes (3): CatalogEditor, _segment_in_rectangle(), _Cancellation

### Community 87 - "Legacy Task Runner"
Cohesion: 0.17
Nodes (9): overlaps(), object_footprint(), object_prism(), polygons_overlap(), prepare_polygon(), PreparedPolygon, allowed(), corners() (+1 more)

### Community 89 - "Live Components"
Cohesion: 0.16
Nodes (6): apply_live(), drop_live_seen(), drop_unconfirmed_live(), LiveSettings, LiveWatcher, _xy()

### Community 91 - "Check Onnx Components"
Cohesion: 0.15
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 92 - "Standalone Cube Detection"
Cohesion: 0.15
Nodes (4): _import_piper_vision(), _load_l1(), _check_ranges(), L1Params

### Community 93 - "Run Components"
Cohesion: 0.16
Nodes (3): _ObjState, PhysicsMonitor, _r()

### Community 94 - "Selfcheck Components"
Cohesion: 0.17
Nodes (9): at_observe_pose(), _camera_frame(), camera_pose_for_check(), _check_camera(), check_pick_area(), _pick_area_depth(), pick_area_mask(), _live_camera() (+1 more)

### Community 96 - "Camera Configuration Geometry"
Cohesion: 0.18
Nodes (7): _apply(), _atomic_write(), load_camera_config(), points_base_to_camera(), points_camera_to_base(), save_camera_config(), write_extrinsics()

### Community 97 - "Stack Scene Observation"
Cohesion: 0.19
Nodes (5): _LayerPerception, _observed_levels(), refine_supported_objects(), _support_evidence(), transform_cloud()

### Community 99 - "Selection Components"
Cohesion: 0.25
Nodes (8): clearances(), is_blacklisted(), _radius(), select_target(), dist(), iso_key(), tie(), _xy()

### Community 101 - "Factory Components"
Cohesion: 0.19
Nodes (5): classifier_backend(), make_detector(), make_perception(), use_hint(), config_hints()

### Community 102 - "MuJoCo Scene Construction"
Cohesion: 0.17
Nodes (5): _cfg_get(), require_mujoco(), camera_table(), _geom_min_z(), piper_geometry()

### Community 104 - "Visualize Components"
Cohesion: 0.33
Nodes (5): _draw_base_polygon(), _draw_detection(), draw_result(), label_color(), _text()

### Community 105 - "Motion Ctl Components"
Cohesion: 0.17
Nodes (5): observe_views(), home_target(), jog_target(), parse_targets(), Refused

### Community 106 - "Calib Points Components"
Cohesion: 0.21
Nodes (6): _deg_between(), sanity_report(), table_in_base(), _table_rows(), _unit(), view_geometry()

### Community 107 - "Utils Components"
Cohesion: 0.25
Nodes (4): capture_averaged(), _compatible(), median_depth(), valid_depth_mask()

### Community 108 - "Motion Ctl Components"
Cohesion: 0.20
Nodes (6): build_parser(), _common_parent(), main(), motion_tuning_file(), _open(), _close_mujoco()

### Community 109 - "Catalog Perception Components"
Cohesion: 0.27
Nodes (4): _center_round_fruit(), completeness(), pear_pose(), SandboxPerception

### Community 110 - "Calib Points Components"
Cohesion: 0.22
Nodes (4): transform_points(), pair_arrays(), parse_command(), prediction_error()

### Community 112 - "Recovery Components"
Cohesion: 0.39
Nodes (5): _copy_fields(), model_recovery_context(), _number(), _text(), validate_recovery_decision()

### Community 114 - "Pyagxarm Arm Components"
Cohesion: 0.25
Nodes (5): piper_firmware_tuple(), seamless_teach_switch(), teach_exit_hint(), _arm_fix(), _live_arm()

### Community 115 - "Check Setup Components"
Cohesion: 0.22
Nodes (4): check_config(), check_live(), config_fix(), config_messages()

### Community 116 - "Detector Onnx Components"
Cohesion: 0.43
Nodes (5): _fields(), read_onnx_info(), _text(), _value_info(), _varint()

### Community 126 - "Selfcheck Components"
Cohesion: 0.40
Nodes (3): _point_in_polygon(), _polygons_overlap(), _segments_cross()

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1999 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_perception()` connect `Onsite Components` to `Config Components`, `Pipeline Components`, `Factory Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `TaskRunner` (e.g. with `Detection` and `PerceptionResult`) actually correct?**
  _`TaskRunner` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Teach Poses Components` be split into smaller, more focused modules?**
  _Cohesion score 0.024522383803820928 - nodes in this community are weakly interconnected._
- **Why does `SimObject` connect `Simulation World State` to `Simulation World State`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `Color Calib Components` be split into smaller, more focused modules?**
  _Cohesion score 0.02335308469850122 - nodes in this community are weakly interconnected._
- **Why does `ensure_sim_knn()` connect `Config Components` to `Factory Components`, `Classify Knn Components`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1379 raw AST edges have unresolved endpoints; 6 self-loops; 756 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
