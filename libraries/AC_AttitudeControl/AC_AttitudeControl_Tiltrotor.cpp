#include "AC_AttitudeControl_Tiltrotor.h"
#include <AP_HAL/AP_HAL.h>
#include <AP_Math/AP_Math.h>
#include <AC_PID/AC_PID.h>

#include <stdio.h>


// table of user settable parameters
const AP_Param::GroupInfo AC_AttitudeControl_Tiltrotor::var_info[] = {
    // parameters from parent vehicle
    AP_NESTEDGROUPINFO(AC_AttitudeControl, 0),

    // @Param: RAT_RLL_P
    // @DisplayName: Roll axis rate controller P gain
    // @Description: Roll axis rate controller P gain.  Corrects in proportion to the difference between the desired roll rate vs actual roll rate
    // @Range: 0.01 0.5
    // @Increment: 0.005
    // @User: Standard

    // @Param: RAT_RLL_I
    // @DisplayName: Roll axis rate controller I gain
    // @Description: Roll axis rate controller I gain.  Corrects long-term difference in desired roll rate vs actual roll rate
    // @Range: 0.01 2.0
    // @Increment: 0.01
    // @User: Standard

    // @Param: RAT_RLL_IMAX
    // @DisplayName: Roll axis rate controller I gain maximum
    // @Description: Roll axis rate controller I gain maximum.  Constrains the maximum that the I term will output
    // @Range: 0 1
    // @Increment: 0.01
    // @User: Standard

    // @Param: RAT_RLL_D
    // @DisplayName: Roll axis rate controller D gain
    // @Description: Roll axis rate controller D gain.  Compensates for short-term change in desired roll rate vs actual roll rate
    // @Range: 0.0 0.05
    // @Increment: 0.001
    // @User: Standard

    // @Param: RAT_RLL_FF
    // @DisplayName: Roll axis rate controller feed forward
    // @Description: Roll axis rate controller feed forward
    // @Range: 0 0.5
    // @Increment: 0.001
    // @User: Standard

    // @Param: RAT_RLL_FLTT
    // @DisplayName: Roll axis rate controller target frequency in Hz
    // @Description: Roll axis rate controller target frequency in Hz
    // @Range: 5 100
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_RLL_FLTE
    // @DisplayName: Roll axis rate controller error frequency in Hz
    // @Description: Roll axis rate controller error frequency in Hz
    // @Range: 0 100
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_RLL_FLTD
    // @DisplayName: Roll axis rate controller derivative frequency in Hz
    // @Description: Roll axis rate controller derivative frequency in Hz
    // @Range: 5 100
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_RLL_SMAX
    // @DisplayName: Roll slew rate limit
    // @Description: Sets an upper limit on the slew rate produced by the combined P and D gains. If the amplitude of the control action produced by the rate feedback exceeds this value, then the D+P gain is reduced to respect the limit. This limits the amplitude of high frequency oscillations caused by an excessive gain. The limit should be set to no more than 25% of the actuators maximum slew rate to allow for load effects. Note: The gain will not be reduced to less than 10% of the nominal value. A value of zero will disable this feature.
    // @Range: 0 200
    // @Increment: 0.5
    // @User: Advanced

    // @Param: RAT_RLL_PDMX
    // @DisplayName: Roll axis rate controller PD sum maximum
    // @Description: Roll axis rate controller PD sum maximum.  The maximum/minimum value that the sum of the P and D term can output
    // @Range: 0 1
    // @Increment: 0.01
    // @User: Advanced

    AP_SUBGROUPINFO(_pid_rate_roll, "RAT_RLL_", 1, AC_AttitudeControl_Tiltrotor, AC_PID),

    // @Param: RAT_PIT_P
    // @DisplayName: Pitch axis rate controller P gain
    // @Description: Pitch axis rate controller P gain.  Corrects in proportion to the difference between the desired pitch rate vs actual pitch rate output
    // @Range: 0.01 0.50
    // @Increment: 0.005
    // @User: Standard

    // @Param: RAT_PIT_I
    // @DisplayName: Pitch axis rate controller I gain
    // @Description: Pitch axis rate controller I gain.  Corrects long-term difference in desired pitch rate vs actual pitch rate
    // @Range: 0.01 2.0
    // @Increment: 0.01
    // @User: Standard

    // @Param: RAT_PIT_IMAX
    // @DisplayName: Pitch axis rate controller I gain maximum
    // @Description: Pitch axis rate controller I gain maximum.  Constrains the maximum that the I term will output
    // @Range: 0 1
    // @Increment: 0.01
    // @User: Standard

    // @Param: RAT_PIT_D
    // @DisplayName: Pitch axis rate controller D gain
    // @Description: Pitch axis rate controller D gain.  Compensates for short-term change in desired pitch rate vs actual pitch rate
    // @Range: 0.0 0.05
    // @Increment: 0.001
    // @User: Standard

    // @Param: RAT_PIT_FF
    // @DisplayName: Pitch axis rate controller feed forward
    // @Description: Pitch axis rate controller feed forward
    // @Range: 0 0.5
    // @Increment: 0.001
    // @User: Standard

    // @Param: RAT_PIT_FLTT
    // @DisplayName: Pitch axis rate controller target frequency in Hz
    // @Description: Pitch axis rate controller target frequency in Hz
    // @Range: 5 100
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_PIT_FLTE
    // @DisplayName: Pitch axis rate controller error frequency in Hz
    // @Description: Pitch axis rate controller error frequency in Hz
    // @Range: 0 100
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_PIT_FLTD
    // @DisplayName: Pitch axis rate controller derivative frequency in Hz
    // @Description: Pitch axis rate controller derivative frequency in Hz
    // @Range: 5 100
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_PIT_SMAX
    // @DisplayName: Pitch slew rate limit
    // @Description: Sets an upper limit on the slew rate produced by the combined P and D gains. If the amplitude of the control action produced by the rate feedback exceeds this value, then the D+P gain is reduced to respect the limit. This limits the amplitude of high frequency oscillations caused by an excessive gain. The limit should be set to no more than 25% of the actuators maximum slew rate to allow for load effects. Note: The gain will not be reduced to less than 10% of the nominal value. A value of zero will disable this feature.
    // @Range: 0 200
    // @Increment: 0.5
    // @User: Advanced

    // @Param: RAT_PIT_PDMX
    // @DisplayName: Pitch axis rate controller PD sum maximum
    // @Description: Pitch axis rate controller PD sum maximum.  The maximum/minimum value that the sum of the P and D term can output
    // @Range: 0 1
    // @Increment: 0.01
    // @User: Advanced

    AP_SUBGROUPINFO(_pid_rate_pitch, "RAT_PIT_", 2, AC_AttitudeControl_Tiltrotor, AC_PID),

    // @Param: RAT_YAW_P
    // @DisplayName: Yaw axis rate controller P gain
    // @Description: Yaw axis rate controller P gain.  Corrects in proportion to the difference between the desired yaw rate vs actual yaw rate
    // @Range: 0.10 2.50
    // @Increment: 0.005
    // @User: Standard

    // @Param: RAT_YAW_I
    // @DisplayName: Yaw axis rate controller I gain
    // @Description: Yaw axis rate controller I gain.  Corrects long-term difference in desired yaw rate vs actual yaw rate
    // @Range: 0.010 1.0
    // @Increment: 0.01
    // @User: Standard

    // @Param: RAT_YAW_IMAX
    // @DisplayName: Yaw axis rate controller I gain maximum
    // @Description: Yaw axis rate controller I gain maximum.  Constrains the maximum that the I term will output
    // @Range: 0 1
    // @Increment: 0.01
    // @User: Standard

    // @Param: RAT_YAW_D
    // @DisplayName: Yaw axis rate controller D gain
    // @Description: Yaw axis rate controller D gain.  Compensates for short-term change in desired yaw rate vs actual yaw rate
    // @Range: 0.000 0.02
    // @Increment: 0.001
    // @User: Standard

    // @Param: RAT_YAW_FF
    // @DisplayName: Yaw axis rate controller feed forward
    // @Description: Yaw axis rate controller feed forward
    // @Range: 0 0.5
    // @Increment: 0.001
    // @User: Standard

    // @Param: RAT_YAW_FLTT
    // @DisplayName: Yaw axis rate controller target frequency in Hz
    // @Description: Yaw axis rate controller target frequency in Hz
    // @Range: 1 50
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_YAW_FLTE
    // @DisplayName: Yaw axis rate controller error frequency in Hz
    // @Description: Yaw axis rate controller error frequency in Hz
    // @Range: 0 20
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_YAW_FLTD
    // @DisplayName: Yaw axis rate controller derivative frequency in Hz
    // @Description: Yaw axis rate controller derivative frequency in Hz
    // @Range: 5 50
    // @Increment: 1
    // @Units: Hz
    // @User: Standard

    // @Param: RAT_YAW_SMAX
    // @DisplayName: Yaw slew rate limit
    // @Description: Sets an upper limit on the slew rate produced by the combined P and D gains. If the amplitude of the control action produced by the rate feedback exceeds this value, then the D+P gain is reduced to respect the limit. This limits the amplitude of high frequency oscillations caused by an excessive gain. The limit should be set to no more than 25% of the actuators maximum slew rate to allow for load effects. Note: The gain will not be reduced to less than 10% of the nominal value. A value of zero will disable this feature.
    // @Range: 0 200
    // @Increment: 0.5
    // @User: Advanced

    // @Param: RAT_YAW_PDMX
    // @DisplayName: Yaw axis rate controller PD sum maximum
    // @Description: Yaw axis rate controller PD sum maximum.  The maximum/minimum value that the sum of the P and D term can output
    // @Range: 0 1
    // @Increment: 0.01
    // @User: Advanced

    AP_SUBGROUPINFO(_pid_rate_yaw, "RAT_YAW_", 3, AC_AttitudeControl_Tiltrotor, AC_PID),

    // @Param: THR_MIX_MIN
    // @DisplayName: Throttle Mix Minimum
    // @Description: Throttle vs attitude control prioritisation used when landing (higher values mean we prioritise attitude control over throttle)
    // @Range: 0.1 0.25
    // @User: Advanced
    AP_GROUPINFO("THR_MIX_MIN", 4, AC_AttitudeControl_Tiltrotor, _thr_mix_min, AC_ATTITUDE_CONTROL_MIN_DEFAULT),

    // @Param: THR_MIX_MAX
    // @DisplayName: Throttle Mix Maximum
    // @Description: Throttle vs attitude control prioritisation used during active flight (higher values mean we prioritise attitude control over throttle)
    // @Range: 0.5 0.9
    // @User: Advanced
    AP_GROUPINFO("THR_MIX_MAX", 5, AC_AttitudeControl_Tiltrotor, _thr_mix_max, AC_ATTITUDE_CONTROL_MAX_DEFAULT),

    // @Param: THR_MIX_MAN
    // @DisplayName: Throttle Mix Manual
    // @Description: Throttle vs attitude control prioritisation used during manual flight (higher values mean we prioritise attitude control over throttle)
    // @Range: 0.1 0.9
    // @User: Advanced
    AP_GROUPINFO("THR_MIX_MAN", 6, AC_AttitudeControl_Tiltrotor, _thr_mix_man, AC_ATTITUDE_CONTROL_MAN_DEFAULT),

    // @Param: THR_G_BOOST
    // @DisplayName: Throttle-gain boost
    // @Description: Throttle-gain boost ratio. A value of 0 means no boosting is applied, a value of 1 means full boosting is applied. Describes the ratio increase that is applied to angle P and PD on pitch and roll.
    // @Range: 0 1
    // @User: Advanced
    AP_GROUPINFO("THR_G_BOOST", 7, AC_AttitudeControl_Tiltrotor, _throttle_gain_boost, 0.0f),

    AP_GROUPEND
};

// run lowest level body-frame rate controller and send outputs to the motors
void AC_AttitudeControl_Tiltrotor::rate_controller_run() {

    if (tiltrotor_5dof_enabled)
    {

        const RC_Channel *pitch_ch = rc().find_channel_for_option(RC_Channel::AUX_FUNC::MOUNT2_PITCH);
        const float pitch_in = (pitch_ch == nullptr) ? 0.0f : pitch_ch->norm_input();
        pitch_offset_deg = 20.0f * pitch_in;

        virtual_pitch_offset_deg = target_virtual_pitch;
        _motors.set_pitch_angle(pitch_offset_deg - virtual_pitch_offset_deg);
        Vector3f trim;
        trim.x = 0;
        trim.z = 0;
        trim.y = ToRad(pitch_offset_deg - virtual_pitch_offset_deg);
        AP::ahrs().set_trim(trim);
    }
    else
    {
        _motors.set_pitch_angle(0.0f);
        Vector3f trim;
        trim.x = 0;
        trim.z = 0;
        trim.y = 0;
        AP::ahrs().set_trim(trim);

        // put these all back at zero so no jump transitioning into 5dof
        pitch_offset_deg = 0.0f;
        virtual_pitch_offset_deg = 0.0f;
        target_virtual_pitch = 0.0f;
    }

#if CONFIG_HAL_BOARD == HAL_BOARD_SITL
    printf("%f %f %f\n", degrees(AP::ahrs().get_pitch()), pitch_offset_deg, virtual_pitch_offset_deg);
#endif

    AC_AttitudeControl_Multi::rate_controller_run();
}



// Command an euler roll and pitch angle and an euler yaw rate with angular velocity feedforward and smoothing
void AC_AttitudeControl_Tiltrotor::input_euler_angle_roll_pitch_euler_rate_yaw(float euler_roll_angle_cd, float euler_pitch_angle_cd, float euler_yaw_rate_cds) {
    set_virtual_pitch(euler_pitch_angle_cd / 100.0f);
    float pitch_target = tiltrotor_5dof_enabled ? 0.0f : euler_pitch_angle_cd;
    pitch_target = euler_pitch_angle_cd;

    AC_AttitudeControl_Multi::input_euler_angle_roll_pitch_euler_rate_yaw(euler_roll_angle_cd, pitch_target, euler_yaw_rate_cds);
}

// Command an euler roll, pitch and yaw angle with angular velocity feedforward and smoothing
void AC_AttitudeControl_Tiltrotor::input_euler_angle_roll_pitch_yaw(float euler_roll_angle_cd, float euler_pitch_angle_cd, float euler_yaw_angle_cd, bool slew_yaw) {
    set_virtual_pitch(euler_pitch_angle_cd / 100.0f);
    float pitch_target = tiltrotor_5dof_enabled ? 0.0f : euler_pitch_angle_cd;
    pitch_target = euler_pitch_angle_cd;

    AC_AttitudeControl_Multi::input_euler_angle_roll_pitch_yaw(euler_roll_angle_cd, pitch_target, euler_yaw_angle_cd, slew_yaw);
}

// Command a thrust vector and heading rate
void AC_AttitudeControl_Tiltrotor::input_thrust_vector_rate_heading(const Vector3f& thrust_vector, float heading_rate_cds, bool slew_yaw)
{
    // convert thrust vector to a roll and pitch angles
    // this negates the advantage of using thrust vector control, but works just fine
    Vector3f angle_target = attitude_from_thrust_vector(thrust_vector, _ahrs.yaw).to_vector312();

    set_virtual_pitch(degrees(angle_target.y));
    float pitch_target = tiltrotor_5dof_enabled ? 0.0f : (degrees(angle_target.y) * 100.0f);
    pitch_target = (degrees(angle_target.y) * 100.0f);

    input_euler_angle_roll_pitch_euler_rate_yaw(degrees(angle_target.x) * 100.0f, pitch_target, heading_rate_cds);
}

// Command a thrust vector, heading and heading rate
void AC_AttitudeControl_Tiltrotor::input_thrust_vector_heading(const Vector3f& thrust_vector, float heading_angle_cd, float heading_rate_cds)
{
    // convert thrust vector to a roll and pitch angles
    Vector3f angle_target = attitude_from_thrust_vector(thrust_vector, _ahrs.yaw).to_vector312();

    set_virtual_pitch(degrees(angle_target.y));
    float pitch_target = tiltrotor_5dof_enabled ? 0.0f : (degrees(angle_target.y) * 100.0f);
    pitch_target = (degrees(angle_target.y) * 100.0f);

    // note that we are throwing away heading rate here
    input_euler_angle_roll_pitch_yaw(degrees(angle_target.x) * 100.0f, pitch_target, heading_angle_cd, true);
}

/*
    all other input functions should zero thrust vectoring
*/

// Command euler yaw rate and pitch angle with roll angle specified in body frame
// (used only by tailsitter quadplanes)
void AC_AttitudeControl_Tiltrotor::input_euler_rate_yaw_euler_angle_pitch_bf_roll(bool plane_controls, float euler_roll_angle_cd, float euler_pitch_angle_cd, float euler_yaw_rate_cds) {
    set_pitch_offset(0.0f);
    set_virtual_pitch(0.0f);

    AC_AttitudeControl_Multi::input_euler_rate_yaw_euler_angle_pitch_bf_roll(plane_controls, euler_roll_angle_cd, euler_pitch_angle_cd, euler_yaw_rate_cds);
}

// Command an euler roll, pitch, and yaw rate with angular velocity feedforward and smoothing
void AC_AttitudeControl_Tiltrotor::input_euler_rate_roll_pitch_yaw(float euler_roll_rate_cds, float euler_pitch_rate_cds, float euler_yaw_rate_cds) {
    set_pitch_offset(0.0f);
    set_virtual_pitch(0.0f);

    AC_AttitudeControl_Multi::input_euler_rate_roll_pitch_yaw(euler_roll_rate_cds, euler_pitch_rate_cds, euler_yaw_rate_cds);
}

// Command an angular velocity with angular velocity feedforward and smoothing
void AC_AttitudeControl_Tiltrotor::input_rate_bf_roll_pitch_yaw(float roll_rate_bf_cds, float pitch_rate_bf_cds, float yaw_rate_bf_cds) {
    set_pitch_offset(0.0f);
    set_virtual_pitch(0.0f);

    AC_AttitudeControl_Multi::input_rate_bf_roll_pitch_yaw(roll_rate_bf_cds, pitch_rate_bf_cds, yaw_rate_bf_cds);
}

// Command an angular velocity with angular velocity feedforward and smoothing
void AC_AttitudeControl_Tiltrotor::input_rate_bf_roll_pitch_yaw_2(float roll_rate_bf_cds, float pitch_rate_bf_cds, float yaw_rate_bf_cds) {
    set_pitch_offset(0.0f);
    set_virtual_pitch(0.0f);

    AC_AttitudeControl_Multi::input_rate_bf_roll_pitch_yaw_2(roll_rate_bf_cds, pitch_rate_bf_cds, yaw_rate_bf_cds);
}

// Command an angular velocity with angular velocity smoothing using rate loops only with integrated rate error stabilization
void AC_AttitudeControl_Tiltrotor::input_rate_bf_roll_pitch_yaw_3(float roll_rate_bf_cds, float pitch_rate_bf_cds, float yaw_rate_bf_cds) {
    set_pitch_offset(0.0f);
    set_virtual_pitch(0.0f);

    AC_AttitudeControl_Multi::input_rate_bf_roll_pitch_yaw_3(roll_rate_bf_cds, pitch_rate_bf_cds, yaw_rate_bf_cds);
}

// Command an angular step (i.e change) in body frame angle
void AC_AttitudeControl_Tiltrotor::input_angle_step_bf_roll_pitch_yaw(float roll_angle_step_bf_cd, float pitch_angle_step_bf_cd, float yaw_angle_step_bf_cd) {
    set_pitch_offset(0.0f);
    set_virtual_pitch(0.0f);

    AC_AttitudeControl_Multi::input_angle_step_bf_roll_pitch_yaw(roll_angle_step_bf_cd, pitch_angle_step_bf_cd, yaw_angle_step_bf_cd);
}

// Command a Quaternion attitude with feedforward and smoothing
// attitude_desired_quat: is updated on each time_step (_dt) by the integral of the angular velocity
// not used anywhere in current code, panic in SITL so this implementation is not overlooked
void AC_AttitudeControl_Tiltrotor::input_quaternion(Quaternion& attitude_desired_quat, Vector3f ang_vel_target) {
#if CONFIG_HAL_BOARD == HAL_BOARD_SITL
    AP_HAL::panic("input_quaternion not implemented AC_AttitudeControl_Tiltrotor");
#endif

    set_pitch_offset(0.0f);
    set_virtual_pitch(0.0f);

    AC_AttitudeControl_Multi::input_quaternion(attitude_desired_quat, ang_vel_target);
}

AC_AttitudeControl_Tiltrotor *AC_AttitudeControl_Tiltrotor::_singleton = nullptr;