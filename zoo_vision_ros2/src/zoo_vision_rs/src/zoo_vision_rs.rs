use std::env;

use anyhow::{Error, Result};
use std::thread;

const STACK_SIZE: usize = 10 * 15 * 1024 * 1024;

fn run() -> Result<(), Error> {
    // Usual main code goes here
    let context = rclrs::Context::new(env::args())?;

    let node = rclrs::create_node(&context, "rerun_forwarder_rs")?;

    let mut num_messages: usize = 0;
    println!("Listening from rust...");
    // let _subscription = node.create_subscription::<zoo_msgs::msg::Image12m, _>(
    //     "/input_camera/image",
    //     rclrs::QOS_PROFILE_DEFAULT,
    //     move |msg: rclrs::ReadOnlyLoanedMessage<'_, zoo_msgs::msg::Image12m>| {
    //         num_messages += 1;
    //         // let frame_id = String::from_utf8_lossy(msg.header.frame_id.data.as_slice());
    //         // println!("I heard: '{}'", frame_id);
    //         println!("(Got {} messages so far)", num_messages);
    //     },
    // )?;

    let _subscription = node.create_subscription::<zoo_msgs::msg::rmw::Image12m, _>(
        "/input_camera/image",
        rclrs::QOS_PROFILE_SENSOR_DATA,
        move |msg: rclrs::ReadOnlyLoanedMessage<'_, zoo_msgs::msg::rmw::Image12m>| {
            num_messages += 1;
            let frame_id = String::from_utf8_lossy(msg.header.frame_id.data.as_slice());
            println!("I heard: '{}'", frame_id);
            // println!("(Got {} messages so far)", num_messages);
        },
    )?;
    rclrs::spin(node).map_err(|err| err.into())
}

fn main() -> Result<(), Error> {
    // Spawn thread with explicit stack size
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    return child.join().unwrap();
}
