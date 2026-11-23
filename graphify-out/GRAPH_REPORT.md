# Graph Report - 智能车项目  (2026-11-23)

## Corpus Check
- Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.

## Summary
- 4959 nodes · 13931 edges · 160 communities (103 shown, 57 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 744 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Teach Poses Components
- Teach Poses Components
- Calib Points Components
- Scenarios Components
-  Components
- Calib Verify Components
- Simulation World State
- Preview Simulation Session
- Onsite Components
- Browser Task Controls
- Segment Components
- Pyagxarm Arm Components
- MuJoCo Arm Control
- Pipeline Components
- Legacy Motion Skills
- Camera Configuration Geometry
- Common Components
- Standalone Cube Detection
- Teach Record Components
- Classify Knn Components
- Teach Classes Components
- Camera Handeye Calibration
- Language Model Planning
- Record Components
- Legacy Task Runner
- Export Onnx Components
- Safety Components
- Sim Arm Components
- Geometry Components
- Composable Skill Orchestration
- Placement Region Observation
- Safety Components
- Web Service Routes
- Render Components
- Validated Skill Execution
- Simulation World State
- Legacy Task Runner
- Simulation World State
- Ik Components
- Logger Components
- Run Components
- Table Components
- Run Task Components
- Sim Bench Components
- Obstacle Recovery Components
- Views Components
- Coordinate Transform Utilities
- Make Dataset Components
- Kinematics Components
- Selfcheck Components
- Ros Camera Components
- Color Calib Components
- Observation Policy Components
- Pyagxarm Arm Components
- Carried Observation Components
- MuJoCo Scene Construction
- Features Components
- Extrinsics Components
- MuJoCo Scene Construction
- Motion Ctl Components
- Skill Protocol Components
- Sequential Batch Planning
- Preview Simulation Session
- Legacy Task Runner
- Color Calib Components
- Grasp Components
- Detector Onnx Components
- MuJoCo Physics Engine
- Check Setup Components
- Grasp Components
- Capture Components
- Reach Components
- Camera Configuration Geometry
- Color Calib Components
- Grasp Components
- Catalog Components
- Factory Components
- Operator Components
- Calib Points Components
- Teach Classes Components
- Interactive Physics Session
- Color Calib Components
- Selection Components
- Color Calib Components
- Ui Components
- Browser Scene Editor
- Video Components
- Grasp Components
- Teach Classes Components
- Detector Onnx Components
- File Camera Components
- Video Components
- Detector Onnx Components
- View Components
- Grasp Components
- Teach Classes Components
- Sim Camera Components
- Assistant Bench Components
- Ros Camera Components
- Extrinsics Components
- Moveit Bridge Components
- Check Setup Components
- Teach Classes Components
- Check Onnx Components
- Standalone Cube Detection
- Destinations Components
- Preview Components
- MuJoCo Scene Construction
- Selfcheck Components
- Registration Components
- Motion Ctl Components
- MuJoCo Scene Construction
- Color Calib Components
- Color Calib Components
- Teach Classes Components
- Pyagxarm Arm Components
- Live Components
- Onsite Components
- Live Components
- Web Service Routes
- Pyagxarm Arm Components
- Check Setup Components
- Dialogue Context Management
- Run Components
- Sim Bench Components
- Sim Bench Components
- Teach Classes Components
- Assistant Ui Components
- Simulation World State
- Task Contract Components
- Selfcheck Components
-   Main   Components
- Carried Observation Components
- Sim Arm Components
- MuJoCo Arm Control
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

## Communities (160 total, 57 thin omitted)

### Community 0 - "Teach Poses Components"
Cohesion: 0.03
Nodes (36): parse_zone(), operator_action(), apply_changes(), ccw_polygon(), _Delete, distance_to_polygon(), effective(), _finite_number() (+28 more)

### Community 1 - "Teach Poses Components"
Cohesion: 0.03
Nodes (45): build_parser(), check_args(), main(), build_parser(), check_args(), main(), build_parser(), add_common_args() (+37 more)

### Community 2 - "Calib Points Components"
Cohesion: 0.03
Nodes (31): _as_array(), _as_list(), _Calib, check_point_geometry(), clean_depth(), compare_extrinsics(), _copy_or_none(), _deg_between() (+23 more)

### Community 3 - "Scenarios Components"
Cohesion: 0.04
Nodes (54): g(), _arc_capsules(), _as_list(), _axis_angle(), _banana_parts(), _catalog_objects(), class_grasp(), default_intrinsics() (+46 more)

### Community 4 - " Components"
Cohesion: 0.04
Nodes (21): placement_metadata(), _Cancellation, _ViewerClock, make_camera(), _make_sim_camera(), _pose(), capture_averaged(), _compatible() (+13 more)

### Community 5 - "Calib Verify Components"
Cohesion: 0.03
Nodes (37): residual_table(), approach_z(), corrected_extrinsics(), corrected_transform(), correction_formula(), _deviations(), _fmt_d(), hover_plan() (+29 more)

### Community 6 - "Simulation World State"
Cohesion: 0.04
Nodes (11): _check_graspable(), _class_grasp(), compound_sections(), find_grasp(), find_top_down_grasp(), _no_grasp_reason(), _require_grasps(), _dome_taper_deg() (+3 more)

### Community 7 - "Preview Simulation Session"
Cohesion: 0.04
Nodes (4): destination_label(), forget_inspection(), SimulationSession, via()

### Community 8 - "Onsite Components"
Cohesion: 0.04
Nodes (31): aruco_dictionary(), pairs_from_json(), ArmPose, average_poses(), backup_file(), backup_path(), coverage_check(), dotted_keys() (+23 more)

### Community 9 - "Browser Task Controls"
Cohesion: 0.08
Nodes (81): add(), api(), assistantText(), box(), cameraRefreshLoop(), canvas, colorKey(), context (+73 more)

### Community 10 - "Segment Components"
Cohesion: 0.05
Nodes (37): _side_support_evidence(), _support_evidence(), points_in_zones(), _barriers(), cap_band(), cap_points(), completed_extents(), _crop() (+29 more)

### Community 11 - "Pyagxarm Arm Components"
Cohesion: 0.06
Nodes (6): ArmError, Stamped, _import_sdk(), _num(), PyAgxArm, zero_pose_problem()

### Community 12 - "MuJoCo Arm Control"
Cohesion: 0.05
Nodes (4): _cfg_vec(), MujocoArm, _Profile, _Trajectory

### Community 13 - "Pipeline Components"
Cohesion: 0.07
Nodes (20): _box_side_surfaces(), _LayerPerception, _observed_levels(), refine_supported_objects(), Detector2D, InstanceClassifier, Detection, Frame (+12 more)

### Community 14 - "Legacy Motion Skills"
Cohesion: 0.09
Nodes (14): rotation_angle(), GraspPlan, PickResult, PlacePlan, linear_moves_enabled(), free_tilt_limit(), _force_value(), gripper_force_for() (+6 more)

### Community 15 - "Camera Configuration Geometry"
Cohesion: 0.06
Nodes (28): _apply(), _as_matrix(), _atomic_write(), camera_pose_in_base(), _check_rigid(), Extrinsics, extrinsics_from_dict(), is_optical_frame() (+20 more)

### Community 16 - "Common Components"
Cohesion: 0.06
Nodes (22): Config, deep_merge(), _find_root(), _parse_value(), read_yaml(), _set_dotted(), write_yaml(), auto_due() (+14 more)

### Community 17 - "Standalone Cube Detection"
Cohesion: 0.06
Nodes (17): Intrinsics, cmd_l1(), load_frame(), load_image_pair(), RgbdFrame, color_mask(), CubeDetection, depth_to_base() (+9 more)

### Community 18 - "Teach Record Components"
Cohesion: 0.06
Nodes (23): depth_to_mm(), _imwrite_atomic(), save_frame(), atomic_write_text(), run_review(), contact_sheet(), crop_of(), instance_records() (+15 more)

### Community 19 - "Classify Knn Components"
Cohesion: 0.05
Nodes (16): catalog_entries(), _cache_path(), _center_round_fruit(), _classifier(), _ColorSeededKnn, completeness(), configure_sandbox(), make_sandbox_perception() (+8 more)

### Community 20 - "Teach Classes Components"
Cohesion: 0.05
Nodes (22): list_frames(), accepted_labels(), add_sim_ground_truth(), Batch, check_label(), clean_label(), config_hints(), confirm_labels() (+14 more)

### Community 21 - "Camera Handeye Calibration"
Cohesion: 0.06
Nodes (22): build_parser(), cmd_board(), cmd_handeye(), cmd_show(), cmd_transform(), _collect_offline(), _default(), _load_cam() (+14 more)

### Community 22 - "Language Model Planning"
Cohesion: 0.05
Nodes (20): _CancellableStream, _iter_sse(), _json_object(), LLMError, LLMSettings, _model_placement(), number(), pair() (+12 more)

### Community 23 - "Record Components"
Cohesion: 0.06
Nodes (19): CameraConfig, cmd_record(), _collect_live(), ArmPoseReader, _image_to_numpy(), RosRgbd, _atomic_imwrite(), _atomic_text() (+11 more)

### Community 25 - "Export Onnx Components"
Cohesion: 0.06
Nodes (23): run_cmd(), box_iou(), build_parser(), compare_onnx_outputs(), default_images(), export(), main(), match_detections() (+15 more)

### Community 26 - "Safety Components"
Cohesion: 0.06
Nodes (4): MotionResult, _rejected(), SafeArm, SafetyGuard

### Community 28 - "Geometry Components"
Cohesion: 0.07
Nodes (21): CameraIntrinsics, as_zone_list(), backproject(), check_depth(), check_intrinsics(), densify_polyline(), _dist_array(), has_distortion() (+13 more)

### Community 30 - "Placement Region Observation"
Cohesion: 0.07
Nodes (26): _coverage(), inspect_held(), measurements_changed(), candidate_visibility_filter(), allowed(), _dimensions(), _grip_bend(), _inspection_geometry() (+18 more)

### Community 31 - "Safety Components"
Cohesion: 0.06
Nodes (17): ArmInterface, pose_error(), ArmHealth, flange_to_tcp_from_cfg(), rot_log(), tilt_outward(), canonical_pose6(), _opt() (+9 more)

### Community 32 - "Web Service Routes"
Cohesion: 0.08
Nodes (11): idle_batch(), load_settings(), BusyError, DashboardController, editable_session(), run(), work(), run() (+3 more)

### Community 33 - "Render Components"
Cohesion: 0.09
Nodes (23): execution_waypoints(), resample_waypoints(), _box_prim(), camera_rays(), _check_intrinsics(), _discontinuities(), _gripper_prims(), _hit_box() (+15 more)

### Community 34 - "Validated Skill Execution"
Cohesion: 0.11
Nodes (7): _PlanChanged, SkillExecutor, stationary(), guard(), interrupted(), progress(), search()

### Community 36 - "Legacy Task Runner"
Cohesion: 0.07
Nodes (19): object_prism(), detection_table_z(), polygons_overlap(), prepare_polygon(), PreparedPolygon, release_workspace_reason(), footprint_dims(), held_shape() (+11 more)

### Community 37 - "Simulation World State"
Cohesion: 0.07
Nodes (13): _make_zones(), _box_surface(), _capsule_surface(), _check_color(), _cylinder_surface(), densify_polyline(), _ellipsoid_surface(), fibonacci_sphere() (+5 more)

### Community 38 - "Ik Components"
Cohesion: 0.09
Nodes (4): PhysicsKinematics, _check_pose(), MujocoIK, _wrap_into()

### Community 39 - "Logger Components"
Cohesion: 0.08
Nodes (6): _EventLogger, dumps_compact(), RunLogger, safe_filename(), _short(), to_jsonable()

### Community 40 - "Run Components"
Cohesion: 0.07
Nodes (10): _EventTap, expected_zones(), failure_causes(), final_truth(), _object_cause(), _ObjState, PhysicsMonitor, _pick_cause() (+2 more)

### Community 41 - "Table Components"
Cohesion: 0.07
Nodes (14): TablePlane, parse_polygon(), fit_table_plane(), fit_table_plane_detailed(), _plane_from_points(), _restrict(), TableFit, TableParams (+6 more)

### Community 42 - "Run Task Components"
Cohesion: 0.10
Nodes (26): _attach_mujoco(), _base_summary(), build_parser(), _classifier_text(), _close_log(), _close_mujoco(), _confirm(), _deg() (+18 more)

### Community 43 - "Sim Bench Components"
Cohesion: 0.09
Nodes (21): aggregate(), build_parser(), _cause_key(), failure_report(), fault_point(), format_report(), ground_truth(), is_success() (+13 more)

### Community 44 - "Obstacle Recovery Components"
Cohesion: 0.10
Nodes (27): footprint_corners(), _action(), _ActionFailed, _center(), _destination(), execute_held_recovery(), record(), stopped() (+19 more)

### Community 45 - "Views Components"
Cohesion: 0.08
Nodes (20): _capture_of(), _cloud(), _distances(), merge_observations(), _prefer_pear_neck_views(), _separate_residual(), _uniform(), points_in_polygon() (+12 more)

### Community 46 - "Coordinate Transform Utilities"
Cohesion: 0.10
Nodes (20): closing_angle_of(), fit_rigid(), make_T(), pose6_to_T(), quat_to_rot(), rot_to_quat(), rot_to_rpy(), rot_x() (+12 more)

### Community 47 - "Make Dataset Components"
Cohesion: 0.10
Nodes (16): build_dataset(), fit_plane_ransac(), iter_session(), _load_capture_frame(), main(), box_fn(), ModelBoxes, _parse_roi() (+8 more)

### Community 48 - "Kinematics Components"
Cohesion: 0.10
Nodes (4): _as_joints(), _check_pose(), PiperKinematics, _cfg_float()

### Community 49 - "Selfcheck Components"
Cohesion: 0.10
Nodes (23): arm_self_check(), at_observe_pose(), _check_arm(), _check_assets(), _check_camera(), _check_extrinsics(), _check_reachable(), _check_robot_config() (+15 more)

### Community 50 - "Ros Camera Components"
Cohesion: 0.10
Nodes (4): camera_info_to_intrinsics(), RosCamera, _stamp_to_sec(), CameraError

### Community 52 - "Observation Policy Components"
Cohesion: 0.08
Nodes (15): measurement_visible(), merge_local_detections(), rank_views(), unique_target(), _xy(), GoalVerification, confirmed(), _SourceObservationUnavailable (+7 more)

### Community 54 - "Carried Observation Components"
Cohesion: 0.09
Nodes (17): _compound_core_supported(), mask_carried_depth(), _pose(), _solid_box_source(), _source_surface_supported(), held_anchor(), seating_requires_feedback(), source_anchor() (+9 more)

### Community 55 - "MuJoCo Scene Construction"
Cohesion: 0.14
Nodes (19): _add_environment(), _add_object(), _add_zone(), _as_list(), _camera(), _camera_housing(), _el(), _f() (+11 more)

### Community 56 - "Features Components"
Cohesion: 0.10
Nodes (12): _check_ranges(), _circular_mean_hue(), hsv_ranges_from_samples(), in_ranges(), bgr_to_hsv_pixels(), _crop_mask(), feature_vector(), footprint_shape() (+4 more)

### Community 57 - "Extrinsics Components"
Cohesion: 0.12
Nodes (13): Extrinsics, extrinsics_from_cfg(), extrinsics_from_dict(), is_optical_frame(), _json_file_as_yaml(), _link_to_optical(), load_extrinsics(), non_optical_frame_message() (+5 more)

### Community 58 - "MuJoCo Scene Construction"
Cohesion: 0.09
Nodes (13): assets_dir_for(), build_scene(), _cfg_get(), _compile(), fovy_deg(), _init_arm(), intrinsics_for_fovy(), _make_meta() (+5 more)

### Community 59 - "Motion Ctl Components"
Cohesion: 0.16
Nodes (18): _arm_selfcheck(), cmd_gripper(), cmd_home(), cmd_joints(), cmd_moveit(), cmd_pick(), cmd_status(), cmd_vision() (+10 more)

### Community 60 - "Skill Protocol Components"
Cohesion: 0.15
Nodes (20): plan_limit(), _capabilities(), _destination_summary(), _failure_summary(), _fields(), _final_audit_summary(), _finite(), _goal_worklist_summary() (+12 more)

### Community 61 - "Sequential Batch Planning"
Cohesion: 0.11
Nodes (13): bind_item(), build_items(), completed_position(), _in_completed_zone(), matching_objects(), whole_footprint_at_goal(), whole_footprint_in_zone(), _complete_geometry() (+5 more)

### Community 62 - "Preview Simulation Session"
Cohesion: 0.11
Nodes (6): apply_translation_anchor(), measure(), phase(), progress(), annotate(), camera_pose_in_base()

### Community 63 - "Legacy Task Runner"
Cohesion: 0.11
Nodes (11): held_offset_of(), transit_height(), _footprint(), _has_xy(), _is_stack_base(), line_obstacle_top(), _NullLogger, pick_summary() (+3 more)

### Community 64 - "Color Calib Components"
Cohesion: 0.09
Nodes (14): build_colors_yaml(), circular_mean_hue(), _FlowList, fmt_ranges(), hue_stats(), inside_fraction(), _num(), overlap_fractions() (+6 more)

### Community 65 - "Grasp Components"
Cohesion: 0.12
Nodes (8): neutral_closing_angle(), wrap_pi(), Zone, PlacePlanner, boundary_candidates(), contact_order(), release_slots(), strip_axis()

### Community 66 - "Detector Onnx Components"
Cohesion: 0.13
Nodes (10): build_label_mapping(), _check_metadata(), _check_names(), DetectorError, _first_line(), OnnxInfo, _OpenCVRunner, _probability() (+2 more)

### Community 68 - "Check Setup Components"
Cohesion: 0.12
Nodes (12): build_parser(), camera_topics(), can_up_command(), check_can(), check_modules(), check_ros(), main(), official_can_script() (+4 more)

### Community 69 - "Grasp Components"
Cohesion: 0.13
Nodes (3): choose_closing_angle(), GraspPlanner, _identity_plan()

### Community 70 - "Capture Components"
Cohesion: 0.09
Nodes (7): build_extra(), _Capture, camera_pose_for_frame(), compose_view(), depth_range_auto(), depth_colormap(), side_by_side()

### Community 71 - "Reach Components"
Cohesion: 0.16
Nodes (12): measured_tapered_contact(), supported_grasp(), gap(), accepts_kwarg(), adjust_plan_poses(), _at_z(), _check(), _heights() (+4 more)

### Community 73 - "Color Calib Components"
Cohesion: 0.10
Nodes (12): Command, match_label(), normalize_line(), parse_command(), _parse_margins(), prepare_pose(), same_layout(), _View (+4 more)

### Community 74 - "Grasp Components"
Cohesion: 0.11
Nodes (10): _finite(), _occupied_capsule(), _pad_taper_deg(), section_center_shift(), section_profile(), section_width(), _seg_dist(), table_floor_z() (+2 more)

### Community 75 - "Catalog Components"
Cohesion: 0.13
Nodes (11): CatalogEditor, make_catalog_object(), _number(), restoration_yaw_period(), _segment_in_rectangle(), _xy(), _catalog_object(), _jitter() (+3 more)

### Community 76 - "Factory Components"
Cohesion: 0.13
Nodes (7): classifier_backend(), make_classifier(), make_detector(), make_perception(), task_labels(), _import_piper_vision(), L1NewPerception

### Community 78 - "Calib Points Components"
Cohesion: 0.09
Nodes (6): marker_corners_base(), marker_image(), sim_spots(), sim_target(), sim_touch_pose(), _SimOperator

### Community 79 - "Teach Classes Components"
Cohesion: 0.11
Nodes (10): build_report(), confused_pairs(), confusion_table(), evaluate_samples(), evaluation_text(), knn_params(), _rows(), save_model() (+2 more)

### Community 81 - "Color Calib Components"
Cohesion: 0.10
Nodes (7): build_parser(), calib_header(), compute_ranges(), main(), options_from_args(), RangeOptions, _run()

### Community 82 - "Selection Components"
Cohesion: 0.14
Nodes (8): clearances(), is_blacklisted(), _radius(), select_target(), dist(), iso_key(), tie(), _xy()

### Community 83 - "Color Calib Components"
Cohesion: 0.11
Nodes (3): InstanceSample, SampleBook, weak_instance_warnings()

### Community 85 - "Browser Scene Editor"
Cohesion: 0.25
Nodes (18): buildLibrary(), chooseMode(), clearDestination(), controls(), draw(), drawObject(), drawPlacementPoint(), drawPrimitive() (+10 more)

### Community 87 - "Grasp Components"
Cohesion: 0.11
Nodes (9): expected_table_z(), max_gripper_width(), _required_float(), observe_views(), home_target(), jog_target(), _manual_detection(), parse_targets() (+1 more)

### Community 88 - "Teach Classes Components"
Cohesion: 0.13
Nodes (3): parse_label_command(), plan_labeling(), _Teacher

### Community 89 - "Detector Onnx Components"
Cohesion: 0.14
Nodes (6): decode_output(), letterbox(), nms_per_label(), OnnxDetector, _shape_mismatch_message(), unletterbox_boxes()

### Community 90 - "File Camera Components"
Cohesion: 0.12
Nodes (4): FileCamera, _check_idx(), load_frame(), _name()

### Community 91 - "Video Components"
Cohesion: 0.15
Nodes (3): _LoggerTap, _NullLogger, _PerceptionTap

### Community 92 - "Detector Onnx Components"
Cohesion: 0.17
Nodes (9): _check_image(), _fields(), _onnxruntime_available(), opencv_version(), read_onnx_info(), _text(), to_pixel_box(), _value_info() (+1 more)

### Community 93 - "View Components"
Cohesion: 0.12
Nodes (6): next_index(), flange_to_tcp(), load_extrinsics_or_none(), run_live(), session_dir(), FpsMeter

### Community 94 - "Grasp Components"
Cohesion: 0.16
Nodes (6): capsule_edge_margin(), footprint_capsule(), done(), evaluate(), reserve_ok(), skip()

### Community 95 - "Teach Classes Components"
Cohesion: 0.11
Nodes (4): instance_rows(), layout_of(), layout_unchanged(), rejected_summary()

### Community 97 - "Assistant Bench Components"
Cohesion: 0.20
Nodes (8): BenchSession, main(), random_objects(), run_case(), session_factory(), run_phase(), stacking_objects(), write_json()

### Community 98 - "Ros Camera Components"
Cohesion: 0.18
Nodes (5): check_depth_scale(), depth_msg_to_meters(), fit_intrinsics(), image_msg_to_bgr(), _image_view()

### Community 99 - "Extrinsics Components"
Cohesion: 0.26
Nodes (8): _check_radians(), _numbers(), _parse_transform(), _quat_rot(), _rotation_any(), _tcp_to_flange(), validate_transform(), _vector()

### Community 103 - "Check Onnx Components"
Cohesion: 0.20
Nodes (5): build_parser(), _env(), main(), run(), _synthetic_image()

### Community 104 - "Standalone Cube Detection"
Cohesion: 0.18
Nodes (3): _load_l1(), _check_ranges(), L1Params

### Community 105 - "Destinations Components"
Cohesion: 0.22
Nodes (6): completion_outline(), footprint_in_destination(), _number(), validate_destination(), whole_footprint_at_destination(), object_footprint()

### Community 107 - "MuJoCo Scene Construction"
Cohesion: 0.17
Nodes (5): _cfg_get(), require_mujoco(), camera_table(), _geom_min_z(), piper_geometry()

### Community 108 - "Selfcheck Components"
Cohesion: 0.21
Nodes (7): _camera_frame(), camera_pose_for_check(), check_pick_area(), _pick_area_depth(), pick_area_mask(), _live_camera(), _pick_area_depth()

### Community 109 - "Registration Components"
Cohesion: 0.23
Nodes (5): _check_intr(), _has_dist(), register_depth_to_color(), median_depth(), valid_depth_mask()

### Community 110 - "Motion Ctl Components"
Cohesion: 0.18
Nodes (6): package_root(), build_parser(), _common_parent(), main(), motion_tuning_file(), _open()

### Community 111 - "MuJoCo Scene Construction"
Cohesion: 0.18
Nodes (5): _compound_part(), object_mass(), object_material(), object_material_name(), _object_parts()

### Community 112 - "Color Calib Components"
Cohesion: 0.17
Nodes (5): effective_classes(), missing_label_warnings(), overlap_pairs(), overlap_warnings(), view_warnings()

### Community 113 - "Color Calib Components"
Cohesion: 0.20
Nodes (4): draw_texts(), _eroded_upper_pixels(), hue_distance(), instance_hsv()

### Community 117 - "Onsite Components"
Cohesion: 0.24
Nodes (5): arm_firmware_version(), ensure_motion_ready(), health_of(), take_can_control_confirmed(), _leave_teach_mode()

### Community 120 - "Pyagxarm Arm Components"
Cohesion: 0.25
Nodes (5): T_to_pose6(), seamless_teach_switch(), teach_exit_hint(), _arm_fix(), _live_arm()

### Community 121 - "Check Setup Components"
Cohesion: 0.22
Nodes (4): check_config(), check_live(), config_fix(), config_messages()

### Community 122 - "Dialogue Context Management"
Cohesion: 0.32
Nodes (3): conversation_summary(), model_conversation(), _safe_text()

### Community 129 - "Task Contract Components"
Cohesion: 0.47
Nodes (3): _all_default_request(), bind_task_contract(), _inventory_labels()

### Community 131 - "Selfcheck Components"
Cohesion: 0.40
Nodes (3): _point_in_polygon(), _polygons_overlap(), _segments_cross()

## Knowledge Gaps
- **11 isolated node(s):** `stateLabels`, `labelNames`, `palette`, `ui`, `canvas` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 2035 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **57 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `open_session()` connect `Common Components` to ` Components`, `Logger Components`, `Factory Components`, `Operator Components`, `Clock Components`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `SimulationSession` (e.g. with `CarriedDetections` and `CatalogEditor`) actually correct?**
  _`SimulationSession` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stateLabels`, `labelNames`, `palette` to the rest of the system?**
  _11 weakly-connected nodes found - possible missing edges._
- **Should `Teach Poses Components` be split into smaller, more focused modules?**
  _Cohesion score 0.02865013774104683 - nodes in this community are weakly interconnected._
- **Why does `make_arm()` connect ` Components` to `Pyagxarm Arm Components`, `MuJoCo Arm Control`, `Kinematics Components`, `Safety Components`, `Sim Arm Components`, `Safety Components`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `Teach Poses Components` be split into smaller, more focused modules?**
  _Cohesion score 0.028857381150959132 - nodes in this community are weakly interconnected._
- **Should `Calib Points Components` be split into smaller, more focused modules?**
  _Cohesion score 0.031844929041190725 - nodes in this community are weakly interconnected._
## Scope and interpretation

See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.

Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.

Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.

Health warning: 1414 raw AST edges have unresolved endpoints; 6 self-loops; 769 same-endpoint edges merge in the simple directed graph. These are navigation limits, not confirmed application defects.

graphify-out/ is written into the working tree; review which generated outputs to share before committing.
