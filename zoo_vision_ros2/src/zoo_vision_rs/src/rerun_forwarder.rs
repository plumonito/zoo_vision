use anyhow::{Error, Result};
use rerun::{demo_util::grid, external::glam, external::ndarray};
// use zoo_msgs::msg::rmw::Image12m;

pub struct RerunForwarder {
    recording: rerun::RecordingStream,
}

impl RerunForwarder {
    pub fn new() -> Result<Self, Error> {
        let recording = rerun::RecordingStreamBuilder::new("zoo_vision").serve_web(
            "0.0.0.0",
            Default::default(),
            Default::default(),
            rerun::MemoryLimit::from_bytes(1024 * 1024 * 1024),
            false,
        )?;
        Ok(Self { recording })
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
        msg: &zoo_msgs::msg::rmw::Image12m,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rerun::external::ndarray::ArrayView;
        let data_view = unsafe {
            ArrayView::from_shape_ptr(
                (msg.height as usize, msg.width as usize, 3),
                msg.data.as_ptr(),
            )
        };
        let rr_image =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::BGR, data_view).unwrap();

        self.recording.log("camera0/image", &rr_image)?;
        // println!("Test from forwarder, image id={}", unsafe {
        //     std::str::from_utf8_unchecked(msg.header.frame_id.data.as_slice())
        // });
        Ok(())
    }

    pub fn mask_callback(
        &mut self,
        msg: &zoo_msgs::msg::rmw::Image4m,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rerun::external::ndarray::ArrayView;
        let data_view = unsafe {
            ArrayView::from_shape_ptr((msg.height as usize, msg.width as usize), msg.data.as_ptr())
        };
        let rr_image =
            rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, data_view).unwrap();

        self.recording.log("camera0/image/mask", &rr_image)?;
        // println!("Test from forwarder, mask id={}", unsafe {
        //     std::str::from_utf8_unchecked(msg.header.frame_id.data.as_slice())
        // });
        Ok(())
    }
}
