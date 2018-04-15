import time, rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH     = 0.44704    #  KM


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, wheel_radius,
                 decel_limit, accel_limit, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        min_speed = 0.1   #  KM

        self.Yaw_Controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        throttle_kp           = 1.00     #
        throttle_ki           = 0.0008   #
        throttle_kd           = 0.15     #
        min_throttle_limit    = 0.0      # Minimum throttle value
        max_throttle_limit    = 0.2      # Maximum throttle value

        steer_kp              = 0.30     #
        steer_ki              = 0.01     #
        steer_kd              = 0.00     #
        min_steer_limit       = -max_steer_angle
        max_steer_limit       = max_steer_angle

        self.Throttle_Controller = PID(throttle_kp, throttle_ki, throttle_kd, min_throttle_limit, max_throttle_limit)
        self.Steering_Controller = PID(steer_kp, steer_ki, steer_kd, min_steer_limit, max_steer_limit)

        Tau                   = 0.5      # cut-off frequency = 1/(2*PI*Tau) = 0.32 Hz
        Ts                    = 0.02     # Sample time in sec (50 Hz)

        self.Velocity_LPF     = LowPassFilter(Tau, Ts)

        self.vehicle_mass    = vehicle_mass
        self.fuel_capacity   = fuel_capacity
        self.brake_deadband  = brake_deadband
        self.wheel_radius    = wheel_radius
        self.decel_limit     = decel_limit
        self.accel_limit     = accel_limit

        self.last_time       = rospy.get_time()

    ###########################################################################
    def control(self, target_velocity, target_omega, current_velocity, current_omega, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.Throttle_Controller.reset()
            return 0.0, 0.0, 0.0

        filtered_current_velocity = self.Velocity_LPF.filt(current_velocity)

        # get the open loop steering value
        estimated_steering = self.Yaw_Controller.get_steering(target_velocity,
                                                              target_omega, filtered_current_velocity)
        # steering = self.Yaw_Controller.get_steering(target_velocity, target_omega, current_velocity)

        velocity_error     = target_velocity - filtered_current_velocity
        # velocity_error = target_velocity - current_velocity
        self.last_velocity = filtered_current_velocity
        current_time       = rospy.get_time()
        sample_time        = current_time - self.last_time
        self.last_time     = current_time

        omega_error        = target_omega - current_omega

        throttle = self.Throttle_Controller.step(velocity_error, sample_time)
        brake    = 0.0
        # get the closed loop steering correction value
        steering_correction = self.Steering_Controller.step(omega_error, sample_time)

        steering = estimated_steering + steering_correction

        if (target_velocity == 0.0) and (filtered_current_velocity < 0.1):  # the car is stationary
            throttle = 0.0
            brake    = 400  # to hold the car in place if we stopped are stopped at traffic light
                            # Acceleration = -1 m/s^2
        elif (throttle < 0.1) and (velocity_error < 0.1):                   # the car is decelerating
            throttle = 0.0
            decel    = max(velocity_error, self.decel_limit)                # should be velocity_error / dt
            brake    = abs(decel) * (self.vehicle_mass + GAS_DENSITY * self.fuel_capacity) \
                       * self.wheel_radius                                  # Torque N.m

        if brake < self.brake_deadband:
            brake = 0.0

        return throttle, brake, steering
