use serde_derive::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct ZooConfig {
    pub individuals: HashMap<String, Individual>,
    pub map: Map,
    // pub models: Models,
    pub cameras: HashMap<String, CameraCalib>,
}

#[derive(Deserialize, Debug)]
pub struct Individual {
    pub id: u32,
    pub color: String,
}

#[derive(Deserialize, Debug)]
pub struct Map {
    pub image: String,

    #[serde(rename = "T_map_from_world2")]
    pub t_map_from_world2: [[f32; 3]; 3],
}

// #[derive(Deserialize, Debug)]
// pub struct Models {
//     pub elephant_label_id: u32,
//     pub segmentation: String,
// }

#[derive(Deserialize, Debug)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,

    #[serde(rename = "K")]
    pub k: [[f32; 3]; 3],
}

#[derive(Deserialize, Debug)]
pub struct PoseRt {
    #[serde(rename = "R")]
    pub rotation: [[f32; 3]; 3],
    #[serde(rename = "t")]
    pub translation: [f32; 3],
}

#[derive(Deserialize, Debug)]
pub struct CameraCalib {
    // pub sample_image: String,
    pub intrinsics: CameraIntrinsics,
    #[serde(rename = "T_world_from_camera")]
    pub t_world_from_camera: PoseRt,
}
