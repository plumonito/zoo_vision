use std::env;

use anyhow::{Error, Result};

fn main() -> Result<(), Error> {
    let context = rclrs::Context::new(env::args())?;

    let node = rclrs::create_node(&context, "rerun_forwarder_rs")?;

    // let mut num_messages: usize = 0;

    let _subscription = node.create_subscription::<sensor_msgs::msg::Image, _>(
        "/input_camera/image",
        rclrs::QOS_PROFILE_DEFAULT,
        move |msg: rclrs::ReadOnlyLoanedMessage<'_, sensor_msgs::msg::Image>| {
            // num_messages += 1;
            println!("I heard: '{}'", msg.header.frame_id);
            // println!("(Got {} messages so far)", num_messages);
        },
    )?;

    // let _subscription = node.create_subscription::<sensor_msgs::msg::Image, _>(
    //     "/input_camera/image",
    //     rclrs::QOS_PROFILE_SENSOR_DATA,
    //     move |msg: sensor_msgs::msg::Image| {
    //         num_messages += 1;
    //         println!("I heard: '{}'", msg.header.frame_id);
    //         // println!("(Got {} messages so far)", num_messages);
    //     },
    // )?;
    rclrs::spin(node).map_err(|err| err.into())
}
