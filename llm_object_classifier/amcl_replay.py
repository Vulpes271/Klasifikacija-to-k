"""Deterministic scan-by-scan replay into the existing ROS 2 AMCL executable.

Only map, wheel odometry, scans and ONE common initial pose enter this process.
Ground truth after initialization is deliberately absent from the replay input.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def main():
    import numpy as np
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import LaserScan
    from rosgraph_msgs.msg import Clock
    from builtin_interfaces.msg import Time
    from lifecycle_msgs.srv import ChangeState
    from lifecycle_msgs.msg import Transition
    from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--branch',choices=['A','B'],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.input.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    arrays=np.load(args.input.parent/'scans.npz')
    times=arrays['times'];ranges=arrays['ranges'].copy();odom=arrays['odom']
    if args.branch=='B':ranges[arrays['removed']]=np.nan
    def stamp(t):
        sec=int(t);ns=round((t-sec)*1e9)
        if ns>=1_000_000_000:sec+=1;ns-=1_000_000_000
        return Time(sec=sec,nanosec=ns)
    def xyquat(pose):return math.sin(pose[2]/2),math.cos(pose[2]/2)
    def transform(t,pose):
        msg=TransformStamped();msg.header.stamp=stamp(t);msg.header.frame_id='odom';msg.child_frame_id='base_link'
        msg.transform.translation.x=float(pose[0]);msg.transform.translation.y=float(pose[1])
        msg.transform.rotation.z,msg.transform.rotation.w=xyquat(pose)
        return msg
    rclpy.init();node=rclpy.create_node('pilot_replay')
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
    map_pub=node.create_publisher(OccupancyGrid,'map',qos)
    scan_pub=node.create_publisher(LaserScan,'scan',10)
    clock_pub=node.create_publisher(Clock,'clock',10)
    init_pub=node.create_publisher(PoseWithCovarianceStamped,'initialpose',10)
    broadcaster=TransformBroadcaster(node);static=StaticTransformBroadcaster(node)
    extrinsic=data['scan_transform'];tf=TransformStamped();tf.header.frame_id='base_link';tf.child_frame_id='scan_front_link'
    for axis in 'xyz':setattr(tf.transform.translation,axis,extrinsic['translation'][axis])
    for axis in 'xyzw':setattr(tf.transform.rotation,axis,extrinsic['rotation'][axis])
    static.sendTransform(tf)
    grid=OccupancyGrid();grid.header.frame_id='map';grid.info.resolution=data['map']['resolution']
    grid.info.width=data['map']['width'];grid.info.height=data['map']['height']
    grid.info.origin.position.x=data['map']['origin'][0];grid.info.origin.position.y=data['map']['origin'][1]
    grid.info.origin.orientation.z=math.sin(data['map']['origin'][2]/2)
    grid.info.origin.orientation.w=math.cos(data['map']['origin'][2]/2)
    grid.data=arrays['occupancy'].flatten().tolist()
    received={}
    def callback(msg):
        t=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        pose=msg.pose.pose;q=pose.orientation
        received[round(t,6)]=[pose.position.x,pose.position.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
    sub=node.create_subscription(PoseWithCovarianceStamped,'amcl_pose',callback,qos)
    library=args.output/'amcl_seed.so'
    subprocess.run(['gcc','-shared','-fPIC','-o',str(library),str(Path(__file__).with_name('amcl_seed.c')),'-ldl'],check=True)
    env={**os.environ,'LD_PRELOAD':str(library),'LILOCBENCH_SEED':str(data['seed'])}
    log=(args.output/'amcl.log').open('w')
    process=subprocess.Popen(['/opt/ros/jazzy/lib/nav2_amcl/amcl','--ros-args','--params-file',str(args.input.parent/'amcl.yaml')],env=env,stdout=log,stderr=subprocess.STDOUT)
    def spin_until(predicate,seconds):
        deadline=time.monotonic()+seconds
        while not predicate() and time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('AMCL exited; see amcl.log')
            rclpy.spin_once(node,timeout_sec=0.01)
        return predicate()
    def settle(seconds=0.1):spin_until(lambda:False,seconds)
    def send_clock(t):clock_pub.publish(Clock(clock=stamp(t)))
    try:
        client=node.create_client(ChangeState,'amcl/change_state')
        if not client.wait_for_service(timeout_sec=20):raise RuntimeError('AMCL lifecycle service unavailable')
        for transition in [Transition.TRANSITION_CONFIGURE,Transition.TRANSITION_ACTIVATE]:
            future=client.call_async(ChangeState.Request(transition=Transition(id=transition)))
            if not spin_until(future.done,15) or not future.result().success:raise RuntimeError('AMCL lifecycle transition failed')
        send_clock(float(times[0]));map_pub.publish(grid);broadcaster.sendTransform(transform(float(times[0]),odom[0]));settle(0.5)
        initial=PoseWithCovarianceStamped();initial.header.frame_id='map';initial.header.stamp=stamp(float(times[0]))
        initial.pose.pose.position.x=data['initial_pose'][0];initial.pose.pose.position.y=data['initial_pose'][1]
        initial.pose.pose.orientation.z,initial.pose.pose.orientation.w=xyquat(data['initial_pose'])
        initial.pose.covariance[0]=0.2**2;initial.pose.covariance[7]=0.2**2;initial.pose.covariance[35]=math.radians(10)**2
        init_pub.publish(initial);settle(0.3)
        rows=[]
        for i,t in enumerate(times):
            t=float(t);send_clock(t);broadcaster.sendTransform(transform(t,odom[i]))
            # Make the transform covering the scan end available before the scan arrives.
            broadcaster.sendTransform(transform(t+data['laser']['scan_time'],arrays['odom_end'][i]));settle(0.01)
            msg=LaserScan();msg.header.stamp=stamp(t);msg.header.frame_id='scan_front_link'
            for field in ['angle_min','angle_max','angle_increment','time_increment','scan_time','range_min','range_max']:
                setattr(msg,field,float(data['laser'][field]))
            msg.ranges=ranges[i].astype(np.float32).tolist();scan_pub.publish(msg)
            key=round(t,6);spin_until(lambda:key in received,2 if i==0 else 0.4)
            rows.append({'timestamp':t,'pose':received.get(key)})
            if i%100==0:print(f'{args.branch}: {i+1}/{len(times)} scans, received={len(received)}',flush=True)
        (args.output/'poses.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/'runtime.json').write_text(json.dumps({'amcl_version':'1.3.12','seed':data['seed'],
            'seed_method':'LD_PRELOAD srand48 override; drand48 algorithm unchanged','frames':len(rows),
            'missing':sum(r['pose'] is None for r in rows)},indent=2)+'\n')
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
