#pragma once

/// @file    AC_AttitudeControl_Tiltrotor.h
/// @brief   ArduCopter attitude control library

#include "AC_AttitudeControl.h"
#include "AC_AttitudeControl_Multi.h"
#include <AP_Motors/AP_MotorsMulticopter.h>
#include <GCS_MAVLink/GCS.h>

class AC_AttitudeControl_Tiltrotor : public AC_AttitudeControl_Multi {
public:
    AC_AttitudeControl_Tiltrotor(AP_AHRS_View &ahrs, const AP_MultiCopter &aparm, AP_MotorsMulticopter& motors):
        AC_AttitudeControl_Multi(ahrs,aparm,motors) {

        if (_singleton != nullptr) {
            AP_HAL::panic("Can only be one AC_AttitudeControl_Tiltrotor");
        }
        
        _singleton = this;
    }

    static AC_AttitudeControl_Tiltrotor *get_singleton() {
        return _singleton;
    }

    // user settable parameters
    static const struct AP_Param::GroupInfo var_info[];

    // run lowest level body-frame rate controller and send outputs to the motors
    void rate_controller_run() override;

    // Command a Quaternion attitude with feedforward and smoothing
    // attitude_desired_quat: is updated on each time_step (_dt) by the integral of the angular velocity
    // not used anywhere in current code, panic so this implementation is not overlooked
    void input_quaternion(Quaternion& attitude_desired_quat, Vector3f ang_vel_target) override;
    /*
        override input functions to attitude controller and convert desired angles into thrust angles and substitute for offset angles
    */

    // Command an euler roll and pitch angle and an euler yaw rate with angular velocity feedforward and smoothing
    void input_euler_angle_roll_pitch_euler_rate_yaw(float euler_roll_angle_cd, float euler_pitch_angle_cd, float euler_yaw_rate_cds)  override;

    // Command an euler roll, pitch and yaw angle with angular velocity feedforward and smoothing
    void input_euler_angle_roll_pitch_yaw(float euler_roll_angle_cd, float euler_pitch_angle_cd, float euler_yaw_angle_cd, bool slew_yaw) override;

    // Command a thrust vector in the earth frame and a heading angle and/or rate
    void input_thrust_vector_rate_heading(const Vector3f& thrust_vector, float heading_rate_cds, bool slew_yaw = true) override;
    void input_thrust_vector_heading(const Vector3f& thrust_vector, float heading_angle_cd, float heading_rate_cds) override;

    /*
        all other input functions should zero thrust vectoring and behave as a normal copter
    */

    // Command euler yaw rate and pitch angle with roll angle specified in body frame
    // (used only by tailsitter quadplanes)
    void input_euler_rate_yaw_euler_angle_pitch_bf_roll(bool plane_controls, float euler_roll_angle_cd, float euler_pitch_angle_cd, float euler_yaw_rate_cds) override;

    // Command an euler roll, pitch, and yaw rate with angular velocity feedforward and smoothing
    void input_euler_rate_roll_pitch_yaw(float euler_roll_rate_cds, float euler_pitch_rate_cds, float euler_yaw_rate_cds) override;

     // Command an angular velocity with angular velocity feedforward and smoothing
    void input_rate_bf_roll_pitch_yaw(float roll_rate_bf_cds, float pitch_rate_bf_cds, float yaw_rate_bf_cds) override;

    // Command an angular velocity with angular velocity feedforward and smoothing
    void input_rate_bf_roll_pitch_yaw_2(float roll_rate_bf_cds, float pitch_rate_bf_cds, float yaw_rate_bf_cds) override;

    // Command an angular velocity with angular velocity smoothing using rate loops only with integrated rate error stabilization
    void input_rate_bf_roll_pitch_yaw_3(float roll_rate_bf_cds, float pitch_rate_bf_cds, float yaw_rate_bf_cds) override;

    // Command an angular step (i.e change) in body frame angle
    void input_angle_step_bf_roll_pitch_yaw(float roll_angle_step_bf_cd, float pitch_angle_step_bf_cd, float yaw_angle_step_bf_cd) override;


    void set_pitch_offset(float pitch_deg) {
        pitch_offset_deg = pitch_deg;
    }

    void set_5dof_enable(bool enable)
    {
        if (enable != tiltrotor_5dof_enabled)
        {
            gcs().send_text(MAV_SEVERITY_DEBUG, enable ? "5DoF Enabled" : "5DoF Disabled" );
        }
        tiltrotor_5dof_enabled = enable;
    }


protected:
    void set_virtual_pitch(float target_pitch)
    {
        target_virtual_pitch = target_pitch;
    }

    bool tiltrotor_5dof_enabled;

    // physical pitch offset -> body to horizon
    float pitch_offset_deg; 

    // virtual pitch offset -> thrust vector to horizon
    float target_virtual_pitch;
    float virtual_pitch_offset_deg;


    static AC_AttitudeControl_Tiltrotor *_singleton;
};
