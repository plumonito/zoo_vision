use super::zoo_config::ZooConfig;
use anyhow::{Error, Result};
use image;
use nalgebra::{Matrix3, Matrix4, Vector3};
use ndarray::prelude::*;
use rerun::{demo_util::grid, external::glam, external::ndarray};
use std::{io::Cursor, path::Path};
fn nanosec_from_ros(stamp: &builtin_interfaces::msg::rmw::Time) -> i64 {
    1e9 as i64 * stamp.sec as i64 + stamp.nanosec as i64
}

pub struct RerunForwarder {
    recording: rerun::RecordingStream,
    first_ros_time_ns: Option<i64>,
    // camera_indices: HashMap<String, usize>,
}

fn transform3d_from_2d(t2d: &Matrix3<f32>) -> Matrix4<f32> {
    let mut t3d: Matrix4<f32> = nalgebra::zero();
    for r in [0, 1] {
        for c in [0, 1] {
            t3d[(r, c)] = t2d[(r, c)];
        }
    }
    for r in [0, 1] {
        t3d[(r, 3)] = t2d[(r, 2)];
    }
    t3d[(2, 2)] = 1.0;
    t3d[(3, 3)] = 1.0;
    return t3d;
}

impl RerunForwarder {
    pub fn new(data_path: &Path) -> Result<Self, Error> {
        let recording = rerun::RecordingStreamBuilder::new("zoo_vision").serve_web(
            "0.0.0.0",
            Default::default(),
            Default::default(),
            rerun::MemoryLimit::from_bytes(1024 * 1024 * 1024),
            false,
        )?;

        // Load config
        let file =
            std::fs::File::open(data_path.join("config.json")).expect("Config json file not found");
        let reader = std::io::BufReader::new(file);
        let config: ZooConfig = serde_json::from_reader(reader).expect("Config json not valid");

        let t_map_from_world2 =
            Matrix3::<f32>::from_row_slice(config.map.t_map_from_world2.as_flattened());
        let t_map_from_world = transform3d_from_2d(&t_map_from_world2);

        // Load floor plan
        let map_path = data_path.join(config.map.image);
        println!("Map filename={}", map_path.to_str().unwrap());
        let world_image_rr = rerun::EncodedImage::from_file(map_path)?;
        recording.log_static("world/map", &world_image_rr)?;

        // Map projection
        // {
        //     // let f = [t_map_from_world[(0, 0)], t_map_from_world[(1, 1)]];
        //     // let p = [t_map_from_world[(0, 2)], t_map_from_world[(1, 2)]];
        //     let resolution = [4904.0, 7663.0];
        //     let f = [-1.0, -1.0];
        //     recording.log_static(
        //         "/world/map",
        //         &rerun::Pinhole::from_focal_length_and_resolution(f, resolution)
        //             .with_image_plane_distance(1.0),
        //     )?;
        // }
        let t_world_from_map = t_map_from_world.qr().try_inverse().unwrap();
        let r = t_world_from_map.fixed_view::<3, 3>(0, 0).clone_owned();
        let t = t_world_from_map.fixed_view::<3, 1>(0, 3).clone_owned();
        // t[2] = -1.0;

        recording.log_static(
            "world/map",
            &rerun::Transform3D::from_mat3x3(r.data.0).with_translation(t.data.0[0]),
        )?;

        // Go through config cameras
        const COLUMN_COUNT: u32 = 2;
        const ASPECT_RATIO: f32 = 1.768421053;
        for (index, (camera_name, camera_config)) in config.cameras.iter().enumerate() {
            // Log pinhole in map view
            let resolution = [
                camera_config.intrinsics.width as f32,
                camera_config.intrinsics.height as f32,
            ];
            let f = [
                camera_config.intrinsics.k[0][0],
                camera_config.intrinsics.k[1][1],
            ];
            let p = [
                camera_config.intrinsics.k[0][2],
                camera_config.intrinsics.k[1][2],
            ];
            recording.log_static(
                format!("/world/{}", camera_name),
                &rerun::Pinhole::from_focal_length_and_resolution(f, resolution)
                    .with_principal_point(p)
                    .with_image_plane_distance(3.0),
            )?;

            let r_world_from_camera = Matrix3::<f32>::from_row_slice(
                camera_config.t_world_from_camera.rotation.as_flattened(),
            );
            let t_camera_in_world =
                Vector3::<f32>::from(camera_config.t_world_from_camera.translation);
            recording.log_static(
                format!("/world/{}", camera_name),
                &rerun::Transform3D::from_mat3x3(r_world_from_camera.data.0)
                    .with_translation(t_camera_in_world.data.0[0]),
            )?;

            // Log grid position in 2D view
            let row = index as u32 / COLUMN_COUNT;
            let col = index as u32 % COLUMN_COUNT;
            let t_grid_from_image =
                rerun::Transform3D::from_translation([col as f32, row as f32 / ASPECT_RATIO, 0.0]);
            recording.log_static(format!("/cameras/{}", camera_name), &t_grid_from_image)?;
        }

        // Log an annotation context to assign a label and color to each class
        recording.log_static(
            "/",
            &rerun::AnnotationContext::new([(
                0,
                "Background",
                rerun::Rgba32::from_unmultiplied_rgba(0, 0, 0, 0),
            )]),
        )?;

        Ok(Self {
            recording,
            first_ros_time_ns: None,
            // camera_indices: HashMap::new(),
        })
    }

    pub fn test_me(&mut self, frame_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let points = grid(glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0), 10);
        let colors = grid(glam::Vec3::ZERO, glam::Vec3::splat(255.0), 10)
            .map(|v| rerun::Color::from_rgb(v.x as u8, v.y as u8, v.z as u8));
        self.recording.log(
            "my_points",
            &rerun::Points3D::new(points)
                .with_colors(colors)
                .with_radii([0.5]),
        )?;
        println!("Test from forwarder, frame_id={}", frame_id);
        Ok(())
    }

    pub fn image_callback(
        &mut self,
        camera: &str,
        _channel: &str,
        msg: &zoo_msgs::msg::rmw::Image12m,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let msg_data_slice = unsafe {
            std::slice::from_raw_parts(msg.data.as_ptr(), (msg.height * msg.width * 3) as usize)
        };
        let image = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
            msg.width,
            msg.height,
            msg_data_slice,
        )
        .unwrap();

        // Compress image
        let mut jpg_writer = Cursor::new(Vec::new());
        image.write_to(&mut jpg_writer, image::ImageFormat::Jpeg)?;
        let image_jpg_data = jpg_writer.into_inner();
        let rr_image =
            rerun::EncodedImage::from_file_contents(image_jpg_data).with_media_type("image/jpeg");

        let time_ns = nanosec_from_ros(&msg.header.stamp);
        self.recording.set_time_nanos("ros_time", time_ns);

        let largest_side = if image.width() > image.height() {
            image.width()
        } else {
            image.height()
        };
        let scale = 1.0 / largest_side as f32;
        self.recording.log(
            format!("/cameras/{}/detections", camera),
            &rerun::Transform3D::from_scale(scale),
        )?;
        self.recording.log(
            format!("/cameras/{}/detections", camera),
            &rr_image.with_draw_order(-1.0),
        )?;

        // Clear out detections
        // self.recording.set_time_nanos("ros_time", time_ns - 1);

        // let camera_name = "input_camera";
        // let image_detections_ent = format!("{}/detections", camera_name);
        // self.recording
        //     .log(image_detections_ent, &rerun::Clear::recursive())?;
        // let world_detections_ent = format!("world/{}/detections", camera_name);
        // self.recording
        //     .log(world_detections_ent, &rerun::Clear::recursive())?;
        // println!("Test from forwarder, image id={}", unsafe {
        //     std::str::from_utf8_unchecked(msg.header.frame_id.data.as_slice())
        // });
        Ok(())
    }

    pub fn detection_callback(
        &mut self,
        camera: &str,
        _channel: &str,
        msg: &zoo_msgs::msg::rmw::Detection,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ros_time_ns = nanosec_from_ros(&msg.header.stamp);
        const DROP_SAMPLE_DURATION: i64 = 2 * 1e9 as i64;
        if DROP_SAMPLE_DURATION > 0 {
            match self.first_ros_time_ns {
                Some(t) => {
                    if ros_time_ns < t {
                        return Ok(());
                    }
                }
                None => {
                    self.first_ros_time_ns = Some(ros_time_ns + DROP_SAMPLE_DURATION);
                    return Ok(());
                }
            }
        }

        // Set rerun time
        self.recording
            .set_time_nanos("ros_time", ros_time_ns as i64);

        let detection_count = msg.detection_count as usize;
        let ids: Vec<u16> = (0..detection_count)
            .map(|x| msg.track_ids[x] as u16)
            .collect();

        // Map message data to image array
        assert!(msg.detection_count == msg.masks.sizes[0]);
        let mask_height = msg.masks.sizes[1] as usize;
        let mask_width = msg.masks.sizes[2] as usize;
        let masks: ArrayBase<ndarray::ViewRepr<&u8>, Ix3> = unsafe {
            ArrayView::from_shape_ptr(
                (detection_count, mask_height, mask_width),
                msg.masks.data.as_ptr(),
            )
        };

        let world_positions: ArrayBase<ndarray::ViewRepr<&f32>, Ix2> = unsafe {
            ArrayView::from_shape_ptr((detection_count, 3), msg.world_positions.as_ptr())
        };

        // Log bounding boxes
        let bbox_centers = (0..detection_count).map(|x| msg.bboxes[x].center);
        let bbox_half_sizes = (0..detection_count).map(|x| msg.bboxes[x].half_size);
        self.recording.log(
            format!("/cameras/{}/detections/boxes", camera),
            &rerun::Boxes2D::from_centers_and_half_sizes(bbox_centers, bbox_half_sizes)
                .with_class_ids(ids.clone()),
        )?;

        // Log position in world
        let world_points_rr =
            rerun::Points2D::new(world_positions.axis_iter(Axis(0)).map(|x| (x[0], x[1])));
        self.recording.log(
            format!("/world/detections/{}/positions", camera),
            &world_points_rr.with_class_ids(ids).with_radii([1.0]),
        )?;

        // Log masks
        const THRESHOLD: u8 = (0.8 * 255.0) as u8;
        let mut image_classes = ndarray::Array2::<u8>::zeros((mask_height, mask_width).f());
        for idx in 0..detection_count {
            let track_id = msg.track_ids[idx];
            let mask_i = masks.slice(s![idx, .., ..]);
            for (p, m) in image_classes.iter_mut().zip(mask_i.iter()) {
                if *m >= THRESHOLD {
                    *p = (track_id) as u8;
                }
            }
        }
        let rr_image = rerun::SegmentationImage::try_from(image_classes)?;
        self.recording.log(
            format!("/cameras/{}/detections/masks", camera),
            &rr_image.with_draw_order(1.0).with_opacity(0.7),
        )?;

        // Log processing time
        let processing_time = std::time::Duration::from_nanos(msg.processing_time_ns);
        self.recording.log(
            format!("/processing_times/{}_segmentation", camera),
            &rerun::Scalar::new(processing_time.as_secs_f64() * 1000.0),
        )?;

        Ok(())
    }
}
