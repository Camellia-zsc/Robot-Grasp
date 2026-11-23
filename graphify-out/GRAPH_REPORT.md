# Graph Report - 智能车项目  (2026-11-23)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4959 nodes · 13933 edges · 158 communities (104 shown, 54 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 744 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Onsite Components
- Teach Classes Components
- Calib Points Components
- Simulation World State
- Calib Verify Components
-  Components
- Browser Task Controls
- Onsite Components
- Scenarios Components
- Segment Components
- Pipeline Components
- MuJoCo Arm Control
- Pyagxarm Arm Components
- Legacy Task Runner
- Render Components
- Geometry Components
- Legacy Motion Skills
- Camera Configuration Geometry
- Simulation World State
- Standalone Cube Detection
- Teach Record Components
- Teach Poses Components
- Teach Poses Components
- Language Model Planning
- Classify Color Components
- Camera Handeye Calibration
- Record Components
- Teach Classes Components
- Export Onnx Components
- Preview Simulation Session
- Composable Skill Orchestration
- Safety Components
- Interactive Physics Session
- Sim Camera Components
- Preview Simulation Session
- Validated Skill Execution
- Web Service Routes
- Sim Arm Components
- Simulation World State
- Carried Observation Components
- Safety Components
- Ik Components
- Logger Components
- Run Components
- Run Task Components
- Views Components
- Sim Bench Components
- Make Dataset Components
- Kinematics Components
- Selfcheck Components
- Legacy Task Runner
- Color Calib Components
- Catalog Perception Components
- Preview Simulation Session
- Pyagxarm Arm Components
- Onsite Components
- MuJoCo Scene Construction
- Calib Points Components
- Preview Simulation Session
- Classify Knn Components
- Placement Region Observation
- Coordinate Transform Utilities
- Obstacle Recovery Components
- MuJoCo Scene Construction
- Extrinsics Components
- Motion Ctl Components
- Skill Protocol Components
- Ros Camera Components
- Teach Poses Components
- Catalog Components
- Color Calib Components
- Grasp Components
- Detector Onnx Components
- MuJoCo Physics Engine
- Sequential Batch Planning
- Check Setup Components
- Grasp Components
- Legacy Task Runner
- Held Recovery Components
- Color Calib Components
- Grasp Components
- Legacy Task Runner
- Operator Components
- Ros Camera Components
- Camera Configuration Geometry
- Color Calib Components
- Color Calib Components
- Ui Components
- Browser Scene Editor
- Video Components
- Grasp Components
- Detector Onnx Components
- Reach Components
- Video Components
- Detector Onnx Components
- Grasp Components
- Extrinsics Components
- Interfaces Components
- Assistant Bench Components
- Calib Points Components
- Moveit Bridge Components
- Selection Components
- Check Setup Components
- Check Onnx Components
- Standalone Cube Detection
- Factory Components
- MuJoCo Scene Construction
- Selfcheck Components
- Motion Ctl Components
- MuJoCo Scene Construction
- Color Calib Components
- Color Calib Components
- Pyagxarm Arm Components
- Onsite Components
- Live Components
- Web Service Routes
- Preview Simulation Session
- Pyagxarm Arm Components
- Check Setup Components
- Dialogue Context Management
- Run Components
- Live Components
- Sim Bench Components
- Sim Bench Components
- Assistant Ui Components
- Registration Components
- Simulation World State
- MuJoCo Arm Control
- Selfcheck Components
-   Main   Components
- Carried Observation Components
- Sim Arm Components
- MuJoCo Arm Control
- Validated Skill Execution
- Clock Components
- Detector Onnx Components
- Grasp Components
- Make Dataset Components

## God Nodes (most connected - your core abstractions)
1. `Detection` - 113 edges
2. `SimulationSession` - 87 edges
3. `TaskRunner` - 82 edges
4. `SimObject` - 79 edges
5. `CameraIntrinsics` - 67 edges
6. `MujocoArm` - 62 edges
7. `Frame` - 60 edges
8. `inv_T()` - 55 edges
9. `PyAgxArm` - 51 edges
10. `SafeArm` - 50 edges

## Surprising Connections (you probably didn't know these)
- `inspect_held()` --indirect_call--> `safe_inspection_route()`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/active_observation.py → ws/src/cube_task/cube_task/assistant/placement_observation.py
- `_training_samples()` --uses--> `CameraIntrinsics`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/core/types.py
- `make_sandbox_perception()` --uses--> `Perception`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/catalog_perception.py → ws/src/cube_task/cube_task/perception/pipeline.py
- `DashboardPhysics` --uses--> `SafeArm`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/robot/safety.py
- `DashboardPhysics` --uses--> `SafetyGuard`  [INFERRED]
  ws/src/cube_task/cube_task/assistant/physics.py → ws/src/cube_task/cube_task/robot/safety.py

## Import Cycles
- None detected.

## Communities (158 total, 54 thin omitted)

### Community 0 - "Onsite Components"
Cohesion: 0.04
Nodes (44): next_index(), deep_merge(), _parse_value(), read_yaml(), build_parser(), main(), build_parser(), main() (+36 more)

### Community 1 - "Teach Classes Components"
Cohesion: 0.03
Nodes (36): list_frames(), accepted_labels(), add_sim_ground_truth(), Batch, build_report(), check_label(), clean_label(), config_hints() (+28 more)

### Community 2 - "Calib Points Components"
Cohesion: 0.04
Nodes (36): _as_array(), _as_list(), check_args(), check_point_geometry(), clean_depth(), compare_extrinsics(), _copy_or_none(), _deg_between() (+28 more)

### Community 3 - "Simulation World State"
Cohesion: 0.04
Nodes (16): class_grasp(), _class_grasp(), compound_sections(), find_grasp(), find_top_down_grasp(), _footprint(), fully_visible(), _no_grasp_reason() (+8 more)

### Community 4 - "Calib Verify Components"
Cohesion: 0.04
Nodes (33): approach_z(), check_args(), corrected_extrinsics(), corrected_transform(), _deviations(), _fmt_d(), hover_plan(), HoverPlan (+25 more)

### Community 5 - " Components"
Cohesion: 0.05
Nodes (15): _ViewerClock, capture_averaged(), _compatible(), median_depth(), valid_depth_mask(), SimClock, load_config(), Clock (+7 more)

### Community 6 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 7 - "Onsite Components"
Cohesion: 0.04
Nodes (32): write_yaml(), aruco_dictionary(), correction_formula(), ArmPose, average_poses(), backup_file(), backup_path(), coverage_check() (+24 more)

### Community 8 - "Scenarios Components"
Cohesion: 0.05
Nodes (44): g(), _as_list(), _axis_angle(), _catalog_object(), _catalog_objects(), _check_graspable(), default_intrinsics(), _draw_labels() (+36 more)

### Community 9 - "Segment Components"
Cohesion: 0.06
Nodes (36): _side_support_evidence(), _barriers(), cap_band(), cap_points(), completed_extents(), _crop(), _Ctx, depth_edges() (+28 more)

### Community 10 - "Pipeline Components"
Cohesion: 0.06
Nodes (20): Detector2D, PerceptionResult, TablePlane, parse_zones(), parse_polygon(), points_in_zones(), _import_piper_vision(), L1NewPerception (+12 more)

### Community 11 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (3): _cfg_vec(), MujocoArm, _Trajectory

### Community 12 - "Pyagxarm Arm Components"
Cohesion: 0.06
Nodes (8): ArmError, canonical_pose6(), _import_sdk(), _num(), _opt(), PyAgxArm, _req(), zero_pose_problem()

### Community 13 - "Legacy Task Runner"
Cohesion: 0.06
Nodes (4): TaskReport, _NullLogger, pick_summary(), TaskRunner

### Community 14 - "Render Components"
Cohesion: 0.06
Nodes (32): rot_z(), _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box(), _hit_capsule() (+24 more)

### Community 15 - "Geometry Components"
Cohesion: 0.05
Nodes (27): _box_side_surfaces(), _LayerPerception, _observed_levels(), refine_supported_objects(), _support_evidence(), CameraIntrinsics, as_zone_list(), backproject() (+19 more)

### Community 16 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (14): rotation_angle(), GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), free_tilt_limit(), _force_value(), gripper_force_for() (+6 more)

### Community 17 - "Camera Configuration Geometry"
Cohesion: 0.06
Nodes (28): _apply(), _as_matrix(), _atomic_write(), camera_pose_in_base(), _check_rigid(), Extrinsics, extrinsics_from_dict(), is_optical_frame() (+20 more)

### Community 18 - "Simulation World State"
Cohesion: 0.04
Nodes (9): LivePreview, layout_reach_report(), _release_xys(), _check_color(), _finite_array(), _normalize_part(), _positive(), SimWorld (+1 more)

### Community 19 - "Standalone Cube Detection"
Cohesion: 0.06
Nodes (17): Intrinsics, cmd_l1(), load_frame(), load_image_pair(), RgbdFrame, color_mask(), CubeDetection, depth_to_base() (+9 more)

### Community 20 - "Teach Record Components"
Cohesion: 0.06
Nodes (23): depth_to_mm(), _imwrite_atomic(), save_frame(), atomic_write_text(), run_review(), contact_sheet(), crop_of(), instance_records() (+15 more)

### Community 21 - "Teach Poses Components"
Cohesion: 0.05
Nodes (30): parse_zone(), apply_changes(), ccw_polygon(), _Delete, distance_to_polygon(), effective(), _finite_number(), _fmt_joints() (+22 more)

### Community 22 - "Teach Poses Components"
Cohesion: 0.05
Nodes (11): operator_action(), clip_to_limits(), draw_map(), _floats_or(), joint_limit_violations(), _joint_limits(), map_px(), _Teacher (+3 more)

### Community 23 - "Language Model Planning"
Cohesion: 0.05
Nodes (21): _CancellableStream, _iter_sse(), _json_object(), LLMError, LLMSettings, load_settings(), _model_placement(), number() (+13 more)

### Community 24 - "Classify Color Components"
Cohesion: 0.06
Nodes (21): InstanceClassifier, Detection, Frame, _check_ranges(), _circular_mean_hue(), ColorClassifier, hsv_ranges_from_samples(), in_ranges() (+13 more)

### Community 25 - "Camera Handeye Calibration"
Cohesion: 0.06
Nodes (22): build_parser(), cmd_board(), cmd_handeye(), cmd_show(), cmd_transform(), _collect_offline(), _default(), _load_cam() (+14 more)

### Community 26 - "Record Components"
Cohesion: 0.06
Nodes (19): CameraConfig, cmd_record(), _collect_live(), ArmPoseReader, _image_to_numpy(), RosRgbd, _atomic_imwrite(), _atomic_text() (+11 more)

### Community 27 - "Teach Classes Components"
Cohesion: 0.05
Nodes (8): command_kind(), counts_text(), instance_rows(), layout_unchanged(), parse_label_command(), plan_labeling(), _Teacher, wrap_lines()

### Community 28 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 29 - "Preview Simulation Session"
Cohesion: 0.07
Nodes (15): _coverage(), inspect_held(), measurements_changed(), measurement_visible(), merge_local_detections(), _xy(), inspection_history(), remember_inspection() (+7 more)

### Community 31 - "Safety Components"
Cohesion: 0.06
Nodes (14): ArmInterface, pose_error(), ArmHealth, GripperResult, flange_to_tcp_from_cfg(), tilt_outward(), piper_firmware_tuple(), interpolate_pose() (+6 more)

### Community 32 - "Interactive Physics Session"
Cohesion: 0.06
Nodes (9): DashboardPhysics, Config, _find_root(), _set_dotted(), ensure_sim_knn(), _knn_loadable(), Session, _sim_knn_fingerprint() (+1 more)

### Community 33 - "Sim Camera Components"
Cohesion: 0.05
Nodes (9): make_camera(), _make_sim_camera(), _pose(), FileCamera, _check_idx(), load_frame(), _name(), CameraSource (+1 more)

### Community 34 - "Preview Simulation Session"
Cohesion: 0.05
Nodes (4): footprint_corners(), rank_views(), SimulationSession, via()

### Community 35 - "Validated Skill Execution"
Cohesion: 0.11
Nodes (7): _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search()

### Community 36 - "Web Service Routes"
Cohesion: 0.09
Nodes (10): idle_batch(), BusyError, DashboardController, editable_session(), run(), work(), run(), run() (+2 more)

### Community 39 - "Carried Observation Components"
Cohesion: 0.07
Nodes (22): _compound_core_supported(), mask_carried_depth(), _pose(), _solid_box_source(), _source_surface_supported(), held_anchor(), seating_requires_feedback(), source_anchor() (+14 more)

### Community 40 - "Safety Components"
Cohesion: 0.09
Nodes (4): MotionResult, _rejected(), SafeArm, SafetyGuard

### Community 41 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 42 - "Logger Components"
Cohesion: 0.08
Nodes (6): _EventLogger, dumps_compact(), RunLogger, safe_filename(), _short(), to_jsonable()

### Community 43 - "Run Components"
Cohesion: 0.07
Nodes (10): _EventTap, expected_zones(), failure_causes(), final_truth(), _object_cause(), _ObjState, PhysicsMonitor, _pick_cause() (+2 more)

### Community 44 - "Run Task Components"
Cohesion: 0.10
Nodes (26): _attach_mujoco(), _base_summary(), build_parser(), _classifier_text(), _close_log(), _close_mujoco(), _confirm(), _deg() (+18 more)

### Community 45 - "Views Components"
Cohesion: 0.08
Nodes (21): _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform(), points_in_polygon() (+13 more)

### Community 46 - "Sim Bench Components"
Cohesion: 0.09
Nodes (21): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), format_report(), ground_truth(), is_success() (+13 more)

### Community 47 - "Make Dataset Components"
Cohesion: 0.10
Nodes (16): build_dataset(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes, _parse_roi() (+8 more)

### Community 48 - "Kinematics Components"
Cohesion: 0.10
Nodes (4): _as_joints(), _check_pose(), PiperKinematics, _cfg_float()

### Community 49 - "Selfcheck Components"
Cohesion: 0.10
Nodes (23): arm_self_check(), at_observe_pose(), _check_arm(), _check_assets(), _check_camera(), _check_extrinsics(), _check_reachable(), _check_robot_config() (+15 more)

### Community 50 - "Legacy Task Runner"
Cohesion: 0.10
Nodes (13): _checked_placement(), detection_table_z(), held_offset_of(), transit_height(), _footprint(), footprint_dims(), _has_xy(), held_shape() (+5 more)

### Community 52 - "Catalog Perception Components"
Cohesion: 0.09
Nodes (13): catalog_entries(), _cache_path(), _center_round_fruit(), _classifier(), _ColorSeededKnn, completeness(), configure_sandbox(), make_sandbox_perception() (+5 more)

### Community 55 - "Onsite Components"
Cohesion: 0.07
Nodes (9): camera_pose_for_frame(), compose_view(), depth_range_auto(), flange_to_tcp(), text_panel(), depth_colormap(), side_by_side(), FpsMeter (+1 more)

### Community 56 - "MuJoCo Scene Construction"
Cohesion: 0.14
Nodes (19): _add_environment(), _add_object(), _add_zone(), _as_list(), _camera(), _camera_housing(), _el(), _f() (+11 more)

### Community 57 - "Calib Points Components"
Cohesion: 0.09
Nodes (3): _Calib, measurement_line(), grab_frames()

### Community 58 - "Preview Simulation Session"
Cohesion: 0.09
Nodes (9): destination_label(), unique_target(), GoalVerification, confirmed(), audit(), camera_confirmed(), measured_support(), planning_options() (+1 more)

### Community 59 - "Classify Knn Components"
Cohesion: 0.12
Nodes (3): KnnClassifier, nan_free(), train_knn_from_sim()

### Community 60 - "Placement Region Observation"
Cohesion: 0.11
Nodes (19): _dimensions(), _grip_bend(), _inspection_geometry(), _inspection_visibility_score(), iter_safe_inspection_poses(), ranking(), visibility_score(), JointTransitRoute (+11 more)

### Community 61 - "Coordinate Transform Utilities"
Cohesion: 0.12
Nodes (18): closing_angle_of(), fit_rigid(), make_T(), pose6_to_T(), quat_to_rot(), rot_to_quat(), rot_to_rpy(), rot_x() (+10 more)

### Community 62 - "Obstacle Recovery Components"
Cohesion: 0.10
Nodes (18): completion_outline(), footprint_in_destination(), _number(), validate_destination(), whole_footprint_at_destination(), _angle_error(), _at_origin(), _bind() (+10 more)

### Community 63 - "MuJoCo Scene Construction"
Cohesion: 0.09
Nodes (13): assets_dir_for(), build_scene(), _cfg_get(), _compile(), fovy_deg(), _init_arm(), intrinsics_for_fovy(), _make_meta() (+5 more)

### Community 64 - "Extrinsics Components"
Cohesion: 0.12
Nodes (12): Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml(), load_extrinsics(), non_optical_frame_message(), normalize_mount() (+4 more)

### Community 65 - "Motion Ctl Components"
Cohesion: 0.16
Nodes (18): _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick(), cmd_status(), cmd_vision() (+10 more)

### Community 66 - "Skill Protocol Components"
Cohesion: 0.15
Nodes (20): plan_limit(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _final_audit_summary(), _finite(), _goal_worklist_summary() (+12 more)

### Community 67 - "Ros Camera Components"
Cohesion: 0.11
Nodes (3): camera_info_to_intrinsics(), RosCamera, CameraError

### Community 68 - "Teach Poses Components"
Cohesion: 0.07
Nodes (12): kinematic_arm(), sim_arm(), sim_drag_joints(), sim_drag_tcp(), pose_report(), post_write_report(), reach_report(), reach_text() (+4 more)

### Community 69 - "Catalog Components"
Cohesion: 0.10
Nodes (12): CatalogEditor, make_catalog_object(), _number(), restoration_yaw_period(), _segment_in_rectangle(), _xy(), _arc_capsules(), _banana_parts() (+4 more)

### Community 70 - "Color Calib Components"
Cohesion: 0.09
Nodes (14): build_colors_yaml(), circular_mean_hue(), _FlowList, fmt_ranges(), hue_stats(), inside_fraction(), _num(), overlap_fractions() (+6 more)

### Community 71 - "Grasp Components"
Cohesion: 0.12
Nodes (8): neutral_closing_angle(), wrap_pi(), Zone, PlacePlanner, boundary_candidates(), contact_order(), release_slots(), strip_axis()

### Community 72 - "Detector Onnx Components"
Cohesion: 0.13
Nodes (10): build_label_mapping(), _check_metadata(), _check_names(), DetectorError, _first_line(), OnnxInfo, _OpenCVRunner, _probability() (+2 more)

### Community 74 - "Sequential Batch Planning"
Cohesion: 0.12
Nodes (12): bind_item(), build_items(), completed_position(), _in_completed_zone(), matching_objects(), whole_footprint_at_goal(), whole_footprint_in_zone(), _complete_geometry() (+4 more)

### Community 75 - "Check Setup Components"
Cohesion: 0.12
Nodes (12): build_parser(), camera_topics(), can_up_command(), check_can(), check_modules(), check_ros(), main(), official_can_script() (+4 more)

### Community 76 - "Grasp Components"
Cohesion: 0.13
Nodes (3): choose_closing_angle(), GraspPlanner, _identity_plan()

### Community 77 - "Legacy Task Runner"
Cohesion: 0.11
Nodes (10): adjust_plan_poses(), release_workspace_reason(), _placement_anchor_center(), interrupted(), candidate_allowed(), cell_key(), interrupted(), interruption_reason() (+2 more)

### Community 78 - "Held Recovery Components"
Cohesion: 0.14
Nodes (16): _action(), _ActionFailed, _center(), _destination(), execute_held_recovery(), record(), stopped(), _fresh_target() (+8 more)

### Community 79 - "Color Calib Components"
Cohesion: 0.10
Nodes (12): Command, match_label(), normalize_line(), parse_command(), _parse_margins(), prepare_pose(), same_layout(), _View (+4 more)

### Community 80 - "Grasp Components"
Cohesion: 0.11
Nodes (10): _finite(), _occupied_capsule(), _pad_taper_deg(), section_center_shift(), section_profile(), section_width(), _seg_dist(), table_floor_z() (+2 more)

### Community 81 - "Legacy Task Runner"
Cohesion: 0.13
Nodes (11): candidate_visibility_filter(), allowed(), center_supported(), object_footprint(), object_prism(), polygons_overlap(), prepare_polygon(), PreparedPolygon (+3 more)

### Community 83 - "Ros Camera Components"
Cohesion: 0.15
Nodes (6): check_depth_scale(), depth_msg_to_meters(), fit_intrinsics(), image_msg_to_bgr(), _image_view(), _stamp_to_sec()

### Community 85 - "Color Calib Components"
Cohesion: 0.10
Nodes (7): build_parser(), calib_header(), compute_ranges(), main(), options_from_args(), RangeOptions, _run()

### Community 86 - "Color Calib Components"
Cohesion: 0.11
Nodes (3): InstanceSample, SampleBook, weak_instance_warnings()

### Community 88 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 90 - "Grasp Components"
Cohesion: 0.11
Nodes (9): expected_table_z(), max_gripper_width(), _required_float(), observe_views(), home_target(), jog_target(), _manual_detection(), parse_targets() (+1 more)

### Community 91 - "Detector Onnx Components"
Cohesion: 0.14
Nodes (6): decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), unletterbox_boxes()

### Community 92 - "Reach Components"
Cohesion: 0.21
Nodes (10): supported_grasp(), gap(), accepts_kwarg(), _at_z(), _check(), _heights(), _planning_cancelled(), reach_adjust() (+2 more)

### Community 93 - "Video Components"
Cohesion: 0.15
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 94 - "Detector Onnx Components"
Cohesion: 0.17
Nodes (9): _check_image(), _fields(), _onnxruntime_available(), opencv_version(), read_onnx_info(), _text(), to_pixel_box(), _value_info() (+1 more)

### Community 95 - "Grasp Components"
Cohesion: 0.16
Nodes (6): capsule_edge_margin(), footprint_capsule(), done(), evaluate(), reserve_ok(), skip()

### Community 96 - "Extrinsics Components"
Cohesion: 0.23
Nodes (9): _check_radians(), _link_to_optical(), _numbers(), _parse_transform(), _quat_rot(), _rotation_any(), _tcp_to_flange(), validate_transform() (+1 more)

### Community 98 - "Assistant Bench Components"
Cohesion: 0.20
Nodes (8): BenchSession, main(), random_objects(), run_case(), session_factory(), run_phase(), stacking_objects(), write_json()

### Community 99 - "Calib Points Components"
Cohesion: 0.16
Nodes (6): average_measurements(), BlockParams, fit_plane_ransac(), measure_target_block(), Measurement, _pick_block()

### Community 101 - "Selection Components"
Cohesion: 0.25
Nodes (8): clearances(), is_blacklisted(), _radius(), select_target(), dist(), iso_key(), tie(), _xy()

### Community 103 - "Check Onnx Components"
Cohesion: 0.20
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 104 - "Standalone Cube Detection"
Cohesion: 0.18
Nodes (3): _load_l1(), _check_ranges(), L1Params

### Community 105 - "Factory Components"
Cohesion: 0.28
Nodes (5): classifier_backend(), make_classifier(), make_detector(), make_perception(), task_labels()

### Community 106 - "MuJoCo Scene Construction"
Cohesion: 0.17
Nodes (5): _cfg_get(), require_mujoco(), camera_table(), _geom_min_z(), piper_geometry()

### Community 107 - "Selfcheck Components"
Cohesion: 0.21
Nodes (7): _camera_frame(), camera_pose_for_check(), check_pick_area(), _pick_area_depth(), pick_area_mask(), _live_camera(), _pick_area_depth()

### Community 108 - "Motion Ctl Components"
Cohesion: 0.18
Nodes (6): package_root(), build_parser(), _common_parent(), main(), motion_tuning_file(), _open()

### Community 109 - "MuJoCo Scene Construction"
Cohesion: 0.18
Nodes (5): _compound_part(), object_mass(), object_material(), object_material_name(), _object_parts()

### Community 110 - "Color Calib Components"
Cohesion: 0.17
Nodes (5): effective_classes(), missing_label_warnings(), overlap_pairs(), overlap_warnings(), view_warnings()

### Community 111 - "Color Calib Components"
Cohesion: 0.20
Nodes (4): draw_texts(), _eroded_upper_pixels(), hue_distance(), instance_hsv()

### Community 113 - "Onsite Components"
Cohesion: 0.24
Nodes (5): arm_firmware_version(), ensure_motion_ready(), health_of(), take_can_control_confirmed(), _leave_teach_mode()

### Community 117 - "Pyagxarm Arm Components"
Cohesion: 0.25
Nodes (5): T_to_pose6(), seamless_teach_switch(), teach_exit_hint(), _arm_fix(), _live_arm()

### Community 118 - "Check Setup Components"
Cohesion: 0.22
Nodes (4): check_config(), check_live(), config_fix(), config_messages()

### Community 119 - "Dialogue Context Management"
Cohesion: 0.32
Nodes (3): conversation_summary(), model_conversation(), _safe_text()

### Community 125 - "Registration Components"
Cohesion: 0.38
Nodes (3): _check_intr(), _has_dist(), register_depth_to_color()

### Community 129 - "Selfcheck Components"
Cohesion: 0.40
Nodes (3): _point_in_polygon(), _polygons_overlap(), _segments_cross()

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 2035 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **54 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `open_session()` connect `Onsite Components` to `Interactive Physics Session`, `Sim Camera Components`, ` Components`, `Factory Components`, `Logger Components`, `Clock Components`, `Operator Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `SimulationSession` (e.g. with `CarriedDetections` and `CatalogEditor`) actually correct?**
  _`SimulationSession` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Onsite Components` be split into smaller, more focused modules?**
  _Cohesion score 0.03783783783783784 - nodes in this community are weakly interconnected._
- **Why does `make_arm()` connect ` Components` to `Sim Arm Components`, `Safety Components`, `MuJoCo Arm Control`, `Pyagxarm Arm Components`, `Kinematics Components`, `Safety Components`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `Teach Classes Components` be split into smaller, more focused modules?**
  _Cohesion score 0.03498168498168498 - nodes in this community are weakly interconnected._
- **Should `Calib Points Components` be split into smaller, more focused modules?**
  _Cohesion score 0.03559870550161812 - nodes in this community are weakly interconnected._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1414 raw AST edges have unresolved endpoints; 6 self-loops; 769 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
