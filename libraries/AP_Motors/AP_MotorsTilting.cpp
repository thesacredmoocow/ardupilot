/*
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <AP_Scripting/AP_Scripting_config.h>


// This allows motor roll, pitch, yaw and throttle factors to be changed in flight, allowing vehicle geometry to be changed

#include "AP_MotorsTilting.h"
#include <GCS_MAVLink/GCS.h>
#include <AP_HAL/AP_HAL.h>
#include <SRV_Channel/SRV_Channel.h>

extern const AP_HAL::HAL& hal;

#define debug_print 0



void AP_MotorsTilting::init(motor_frame_class frame_class, motor_frame_type frame_type)
{
    if ((frame_class != motor_frame_class::MOTOR_FRAME_TILTROTOR) || 
        (frame_type != motor_frame_type::MOTOR_FRAME_TYPE_TILTROTOR_X) ||
        (!SRV_Channels::function_assigned(SRV_Channel::k_motor_tilt)) || 
        (!SRV_Channels::function_assigned(SRV_Channel::k_tiltMotorRear)))
    {
        return;
    }

    for(uint8_t i = 0; i < 4; i++)
    {
        motor_enabled[i] = true;
        add_motor_num(i);
        _throttle_factor[i] = 1.0;
    }

    //motor 1: 45
    _test_order[0] = 1;
    _yaw_factor[0] = AP_MOTORS_MATRIX_YAW_FACTOR_CCW;
    _roll_factor[0] = -0.5;
    _pitch_factor[0] = 0.5;

    //motor 2: -135
    _test_order[1] = 3;
    _yaw_factor[1] = AP_MOTORS_MATRIX_YAW_FACTOR_CCW;
    _roll_factor[1] = 0.5;
    _pitch_factor[1] = -0.5;

    //motor 3: -45
    _test_order[2] = 4;
    _yaw_factor[2] = AP_MOTORS_MATRIX_YAW_FACTOR_CW;
    _roll_factor[2] = 0.5;
    _pitch_factor[2] = 0.5;

    //motor 4: 135
    _test_order[3] = 2;
    _yaw_factor[3] = AP_MOTORS_MATRIX_YAW_FACTOR_CW;
    _roll_factor[3] = -0.5;
    _pitch_factor[3] = -0.5;

    _mav_type = MAV_TYPE_QUADROTOR;

    set_initialised_ok(true);
}


void AP_MotorsTilting::load_factors(const factor_table &new_table)
{
    memcpy(_roll_factor,new_table.roll,sizeof(_roll_factor));
    memcpy(_pitch_factor,new_table.pitch,sizeof(_pitch_factor));
    memcpy(_yaw_factor,new_table.yaw,sizeof(_yaw_factor));
    memcpy(_throttle_factor,new_table.throttle,sizeof(_throttle_factor));

#if debug_print
    hal.console->printf("Got new factors:\n");
    for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            hal.console->printf("%i - Roll: %0.2f, Pitch %0.2f, Yaw: %0.2f, throttle %0.2f\n",i,_roll_factor[i],_pitch_factor[i],_yaw_factor[i],_throttle_factor[i]);
        }
    }
#endif

}

// output - sends commands to the motors, 
// Need to take the semaphore to enasure the motor factors are not changed during the mixer calculation
void AP_MotorsTilting::output_to_motors()
{
    const float total_angle = _tiltrotor_max_angle - _tiltrotor_min_angle;
    const float output = (pitch_offset_angle - _tiltrotor_min_angle) / total_angle;
    const uint16_t output_pwm = (uint16_t)(output * 1000.0) + 1000;
    SRV_Channels::set_output_pwm(SRV_Channel::k_motor_tilt, output_pwm);
    SRV_Channels::set_output_pwm(SRV_Channel::k_tiltMotorRear, output_pwm);
    AP_MotorsMatrix::output_to_motors();
}

void AP_MotorsTilting::set_pitch_angle(float angle)
{
    pitch_offset_angle = angle;
}

// output_armed - sends commands to the motors
// includes new scaling stability patch
void AP_MotorsTilting::output_armed_stabilizing()
{
    // apply voltage and air pressure compensation
    const float compensation_gain = thr_lin.get_compensation_gain(); // compensation for battery voltage and altitude

    // pitch thrust input value, +/- 1.0
    const float roll_thrust = (_roll_in + _roll_in_ff) * compensation_gain;

    // pitch thrust input value, +/- 1.0
    const float pitch_thrust = (_pitch_in + _pitch_in_ff) * compensation_gain;

    // yaw thrust input value, +/- 1.0
    float yaw_thrust = (_yaw_in + _yaw_in_ff) * compensation_gain;

    // throttle thrust input value, 0.0 - 1.0
    float throttle_thrust = get_throttle() * compensation_gain;

    // throttle thrust average maximum value, 0.0 - 1.0
    float throttle_avg_max = _throttle_avg_max * compensation_gain;

    // throttle thrust maximum value, 0.0 - 1.0, If thrust boost is active then do not limit maximum thrust
    const float throttle_thrust_max = boost_ratio(1.0, _throttle_thrust_max * compensation_gain);

    // sanity check throttle is above zero and below current limited throttle
    if (throttle_thrust <= 0.0f) {
        throttle_thrust = 0.0f;
        limit.throttle_lower = true;
    }
    if (throttle_thrust >= throttle_thrust_max) {
        throttle_thrust = throttle_thrust_max;
        limit.throttle_upper = true;
    }

    // ensure that throttle_avg_max is between the input throttle and the maximum throttle
    throttle_avg_max = constrain_float(throttle_avg_max, throttle_thrust, throttle_thrust_max);

    // throttle providing maximum roll, pitch and yaw range
    // calculate the highest allowed average thrust that will provide maximum control range
    float throttle_thrust_best_rpy = MIN(0.5f, throttle_avg_max);

    // calculate throttle that gives most possible room for yaw which is the lower of:
    //      1. 0.5f - (rpy_low+rpy_high)/2.0 - this would give the maximum possible margin above the highest motor and below the lowest
    //      2. the higher of:
    //            a) the pilot's throttle input
    //            b) the point _throttle_rpy_mix between the pilot's input throttle and hover-throttle
    //      Situation #2 ensure we never increase the throttle above hover throttle unless the pilot has commanded this.
    //      Situation #2b allows us to raise the throttle above what the pilot commanded but not so far that it would actually cause the copter to rise.
    //      We will choose #1 (the best throttle for yaw control) if that means reducing throttle to the motors (i.e. we favor reducing throttle *because* it provides better yaw control)
    //      We will choose #2 (a mix of pilot and hover throttle) only when the throttle is quite low.  We favor reducing throttle instead of better yaw control because the pilot has commanded it

    // Under the motor lost condition we remove the highest motor output from our calculations and let that motor go greater than 1.0
    // To ensure control and maximum righting performance Hex and Octo have some optimal settings that should be used
    // Y6               : MOT_YAW_HEADROOM = 350, ATC_RAT_RLL_IMAX = 1.0,   ATC_RAT_PIT_IMAX = 1.0,   ATC_RAT_YAW_IMAX = 0.5
    // Octo-Quad (x8) x : MOT_YAW_HEADROOM = 300, ATC_RAT_RLL_IMAX = 0.375, ATC_RAT_PIT_IMAX = 0.375, ATC_RAT_YAW_IMAX = 0.375
    // Octo-Quad (x8) + : MOT_YAW_HEADROOM = 300, ATC_RAT_RLL_IMAX = 0.75,  ATC_RAT_PIT_IMAX = 0.75,  ATC_RAT_YAW_IMAX = 0.375
    // Usable minimums below may result in attitude offsets when motors are lost. Hex aircraft are only marginal and must be handles with care
    // Hex              : MOT_YAW_HEADROOM = 0,   ATC_RAT_RLL_IMAX = 1.0,   ATC_RAT_PIT_IMAX = 1.0,   ATC_RAT_YAW_IMAX = 0.5
    // Octo-Quad (x8) x : MOT_YAW_HEADROOM = 300, ATC_RAT_RLL_IMAX = 0.25,  ATC_RAT_PIT_IMAX = 0.25,  ATC_RAT_YAW_IMAX = 0.25
    // Octo-Quad (x8) + : MOT_YAW_HEADROOM = 300, ATC_RAT_RLL_IMAX = 0.5,   ATC_RAT_PIT_IMAX = 0.5,   ATC_RAT_YAW_IMAX = 0.25
    // Quads cannot make use of motor loss handling because it doesn't have enough degrees of freedom.

    // calculate amount of yaw we can fit into the throttle range
    // this is always equal to or less than the requested yaw from the pilot or rate controller
    float yaw_allowed = 1.0f; // amount of yaw we can fit in
    for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            // calculate the thrust outputs for roll and pitch
            _thrust_rpyt_out[i] = roll_thrust * _roll_factor[i] + pitch_thrust * _pitch_factor[i];

            // Check the maximum yaw control that can be used on this channel
            // Exclude any lost motors if thrust boost is enabled
            if (!is_zero(_yaw_factor[i]) && (!_thrust_boost || i != _motor_lost_index)) {
                const float thrust_rp_best_throttle = throttle_thrust_best_rpy + _thrust_rpyt_out[i];
                float motor_room;
                if (is_positive(yaw_thrust * _yaw_factor[i])) {
                    // room to upper limit
                    motor_room = 1.0 - thrust_rp_best_throttle;
                } else {
                    // room to lower limit
                    motor_room = thrust_rp_best_throttle;
                }
                const float motor_yaw_allowed = MAX(motor_room, 0.0)/fabsf(_yaw_factor[i]);
                yaw_allowed = MIN(yaw_allowed, motor_yaw_allowed);
            }
        }
    }

    // calculate the maximum yaw control that can be used
    // todo: make _yaw_headroom 0 to 1
    float yaw_allowed_min = (float)_yaw_headroom * 0.001f;

    // increase yaw headroom to 50% if thrust boost enabled
    yaw_allowed_min = boost_ratio(0.5, yaw_allowed_min);

    // Let yaw access minimum amount of head room
    yaw_allowed = MAX(yaw_allowed, yaw_allowed_min);

    // Include the lost motor scaled by _thrust_boost_ratio to smoothly transition this motor in and out of the calculation
    if (_thrust_boost && motor_enabled[_motor_lost_index]) {
        // Check the maximum yaw control that can be used on this channel
        // Exclude any lost motors if thrust boost is enabled
        if (!is_zero(_yaw_factor[_motor_lost_index])){
            const float thrust_rp_best_throttle = throttle_thrust_best_rpy + _thrust_rpyt_out[_motor_lost_index];
            float motor_room;
            if (is_positive(yaw_thrust * _yaw_factor[_motor_lost_index])) {
                motor_room = 1.0 - thrust_rp_best_throttle;
            } else {
                motor_room = thrust_rp_best_throttle;
            }
            const float motor_yaw_allowed = MAX(motor_room, 0.0)/fabsf(_yaw_factor[_motor_lost_index]);
            yaw_allowed = boost_ratio(yaw_allowed, MIN(yaw_allowed, motor_yaw_allowed));
        }
    }

    if (fabsf(yaw_thrust) > yaw_allowed) {
        // not all commanded yaw can be used
        yaw_thrust = constrain_float(yaw_thrust, -yaw_allowed, yaw_allowed);
        limit.yaw = true;
    }

    // add yaw control to thrust outputs
    float rpy_low = 1.0f;   // lowest thrust value
    float rpy_high = -1.0f; // highest thrust value
    for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            _thrust_rpyt_out[i] = _thrust_rpyt_out[i] + yaw_thrust * _yaw_factor[i];

            // record lowest roll + pitch + yaw command
            if (_thrust_rpyt_out[i] < rpy_low) {
                rpy_low = _thrust_rpyt_out[i];
            }
            // record highest roll + pitch + yaw command
            // Exclude any lost motors if thrust boost is enabled
            if (_thrust_rpyt_out[i] > rpy_high && (!_thrust_boost || i != _motor_lost_index)) {
                rpy_high = _thrust_rpyt_out[i];
            }
        }
    }
    // Include the lost motor scaled by _thrust_boost_ratio to smoothly transition this motor in and out of the calculation
    if (_thrust_boost) {
        // record highest roll + pitch + yaw command
        if (_thrust_rpyt_out[_motor_lost_index] > rpy_high && motor_enabled[_motor_lost_index]) {
            rpy_high = boost_ratio(rpy_high, _thrust_rpyt_out[_motor_lost_index]);
        }
    }

    // calculate any scaling needed to make the combined thrust outputs fit within the output range
    float rpy_scale = 1.0f;
    if (rpy_high - rpy_low > 1.0f) {
        rpy_scale = 1.0f / (rpy_high - rpy_low);
    }
    if (throttle_avg_max + rpy_low < 0) {
        rpy_scale = MIN(rpy_scale, -throttle_avg_max / rpy_low);
    }

    // calculate how close the motors can come to the desired throttle
    rpy_high *= rpy_scale;
    rpy_low *= rpy_scale;
    throttle_thrust_best_rpy = -rpy_low;
    float thr_adj = throttle_thrust - throttle_thrust_best_rpy;
    if (rpy_scale < 1.0f) {
        // Full range is being used by roll, pitch, and yaw.
        limit.roll = true;
        limit.pitch = true;
        limit.yaw = true;
        if (thr_adj > 0.0f) {
            limit.throttle_upper = true;
        }
        thr_adj = 0.0f;
    } else if (thr_adj < 0.0f) {
        // Throttle can't be reduced to desired value
        // todo: add lower limit flag and ensure it is handled correctly in altitude controller
        thr_adj = 0.0f;
    } else if (thr_adj > 1.0f - (throttle_thrust_best_rpy + rpy_high)) {
        // Throttle can't be increased to desired value
        thr_adj = 1.0f - (throttle_thrust_best_rpy + rpy_high);
        limit.throttle_upper = true;
    }

    // add scaled roll, pitch, constrained yaw and throttle for each motor
    const float throttle_thrust_best_plus_adj = throttle_thrust_best_rpy + thr_adj;
    for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
        if (motor_enabled[i]) {
            _thrust_rpyt_out[i] = (throttle_thrust_best_plus_adj * _throttle_factor[i]) + (rpy_scale * _thrust_rpyt_out[i]);
        }
    }

    // determine throttle thrust for harmonic notch
    // compensation_gain can never be zero
    _throttle_out = throttle_thrust_best_plus_adj / compensation_gain;

    // check for failed motor
    check_for_failed_motor(throttle_thrust_best_plus_adj);
}

// singleton instance
AP_MotorsTilting *AP_MotorsTilting::_singleton;