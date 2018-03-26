import time, rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, brake_deadband, wheel_radius, decel_limit, wheel_base, 
        steer_ratio, max_lat_accel, max_steer_angle, Kp, Ki, Kd):

        min_speed = 1.0 * ONE_MPH
        self.throttle_pid = PID(Kp, Ki, Kd)
        self.yaw_control = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.brake_deadband = brake_deadband
        self.v_mass = vehicle_mass
        self.w_radius = wheel_radius
        self.d_limit = decel_limit
        self.last_time = None
        self.max_speed = 0

    def control(self, target_v, target_omega, current_v, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if self.last_time is None or not dbw_enabled:
            self.last_time = time.time()
            return 0.0, 0.0, 0.0

        if target_v.x > self.max_speed:
            self.max_speed = target_v.x

        dt = time.time() - self.last_time
        error = min(target_v.x, self.max_speed) - current_v.x
        throttle = self.throttle_pid.step(error, dt)
        throttle = max(0.0, min(1.0, throttle)) # Max_throtle = 1.0

        if error < 0: # decelerate
            deceleration        = abs(error) / dt
            if abs(deceleration) > abs(self.d_limit)*500:
                deceleration = self.d_limit*500
            longitudinal_force  = self.v_mass * deceleration
            brake = longitudinal_force * self.w_radius
            if brake < self.brake_deadband:
                brake = 0.0
            throttle = 0.0
        else:
            brake = 0.0
        
        steer = self.yaw_control.get_steering(target_v.x, target_omega.z, current_v.x)
        self.last_time = time.time()

        return throttle, brake, steer

