#include "Copter.h"

#if MODE_LOITER_ENABLED

/*
 * Init and run calls for loiter flight mode
 */

// loiter_init - initialise loiter controller
bool ModeLoiter::init(bool ignore_checks)
{
    if (!copter.failsafe.radio) {
        float target_roll, target_pitch;
        // apply SIMPLE mode transform to pilot inputs
        update_simple_mode();

        // convert pilot input to lean angles
        get_pilot_desired_lean_angles(target_roll, target_pitch, loiter_nav->get_angle_max_cd(), attitude_control->get_althold_lean_angle_max_cd());

        // process pilot's roll and pitch input
        loiter_nav->set_pilot_desired_acceleration(target_roll, target_pitch);
    } else {
        // clear out pilot desired acceleration in case radio failsafe event occurs and we do not switch to RTL for some reason
        loiter_nav->clear_pilot_desired_acceleration();
    }
    loiter_nav->init_target();

    // initialise the vertical position controller
    if (!pos_control->is_active_z()) {
        pos_control->init_z_controller();
    }

    // set vertical speed and acceleration limits
    pos_control->set_max_speed_accel_z(-get_pilot_speed_dn(), g.pilot_speed_up, g.pilot_accel_z);
    pos_control->set_correction_speed_accel_z(-get_pilot_speed_dn(), g.pilot_speed_up, g.pilot_accel_z);

#if AC_PRECLAND_ENABLED
    _precision_loiter_active = false;
    _external_target_ms = 0;
#endif

    return true;
}

#if AC_PRECLAND_ENABLED
void ModeLoiter::set_position_target_yaw_z_from_mavlink(float yaw_cd, float z_cm_neu)
{
    _external_yaw_cd = yaw_cd;
    _external_z_cm = z_cm_neu;
    _external_target_ms = AP_HAL::millis();
}
#endif

#if AC_PRECLAND_ENABLED
bool ModeLoiter::do_precision_loiter()
{
    if (!_precision_loiter_enabled) {
        return false;
    }
    if (copter.ap.land_complete_maybe) {
        return false;        // don't move on the ground
    }
    // if the pilot *really* wants to move the vehicle, let them....
    if (loiter_nav->get_pilot_desired_acceleration().length() > 50.0f) {
        return false;
    }
    if (!copter.precland.target_acquired()) {
        return false; // we don't have a good vector
    }
    return true;
}

void ModeLoiter::precision_loiter_xy()
{
    loiter_nav->clear_pilot_desired_acceleration();
    Vector2f target_pos, target_vel;
    if (!copter.precland.get_target_position_cm(target_pos)) {
        target_pos = inertial_nav.get_position_xy_cm();
    }
    // get the velocity of the target
    copter.precland.get_target_velocity_cms(inertial_nav.get_velocity_xy_cms(), target_vel);

    Vector2f zero;
    Vector2p landing_pos = target_pos.topostype();
    // target vel will remain zero if landing target is stationary
    pos_control->input_pos_vel_accel_xy(landing_pos, target_vel, zero);
    // run pos controller
    pos_control->update_xy_controller();
}
#endif

// loiter_run - runs the loiter controller
// should be called at 100hz or more
void ModeLoiter::run()
{
    float target_roll, target_pitch;
    float target_yaw_rate = 0.0f;
    float target_climb_rate = 0.0f;
    const float yaw_rate_max_cds = attitude_control->get_slew_yaw_max_degs() * 100.0f;

    // set vertical speed and acceleration limits
    pos_control->set_max_speed_accel_z(-get_pilot_speed_dn(), g.pilot_speed_up, g.pilot_accel_z);

    // process pilot inputs unless we are in radio failsafe
    if (!copter.failsafe.radio) {
        // apply SIMPLE mode transform to pilot inputs
        update_simple_mode();

        // convert pilot input to lean angles
        get_pilot_desired_lean_angles(target_roll, target_pitch, loiter_nav->get_angle_max_cd(), attitude_control->get_althold_lean_angle_max_cd());

        // process pilot's roll and pitch input
        loiter_nav->set_pilot_desired_acceleration(target_roll, target_pitch);

        // get pilot's desired yaw rate
        target_yaw_rate = get_pilot_desired_yaw_rate();

        // get pilot desired climb rate
        target_climb_rate = get_pilot_desired_climb_rate(channel_throttle->get_control_in());
        target_climb_rate = constrain_float(target_climb_rate, -get_pilot_speed_dn(), g.pilot_speed_up);
    } else {
        // clear out pilot desired acceleration in case radio failsafe event occurs and we do not switch to RTL for some reason
        loiter_nav->clear_pilot_desired_acceleration();
    }

    // relax loiter target if we might be landed
    if (copter.ap.land_complete_maybe) {
        loiter_nav->soften_for_landing();
    }

    // Loiter State Machine Determination
    AltHoldModeState loiter_state = get_alt_hold_state(target_climb_rate);

    // Loiter State Machine
    switch (loiter_state) {

    case AltHoldModeState::MotorStopped:
        attitude_control->reset_rate_controller_I_terms();
        attitude_control->reset_yaw_target_and_rate();
        pos_control->relax_z_controller(0.0f);   // forces throttle output to decay to zero
        loiter_nav->init_target();
        attitude_control->input_thrust_vector_rate_heading(loiter_nav->get_thrust_vector(), target_yaw_rate, false);
        break;

    case AltHoldModeState::Landed_Ground_Idle:
        attitude_control->reset_yaw_target_and_rate();
        FALLTHROUGH;

    case AltHoldModeState::Landed_Pre_Takeoff:
        attitude_control->reset_rate_controller_I_terms_smoothly();
        loiter_nav->init_target();
        attitude_control->input_thrust_vector_rate_heading(loiter_nav->get_thrust_vector(), target_yaw_rate, false);
        pos_control->relax_z_controller(0.0f);   // forces throttle output to decay to zero
        break;

    case AltHoldModeState::Takeoff:
        // initiate take-off
        if (!takeoff.running()) {
            takeoff.start(constrain_float(g.pilot_takeoff_alt,0.0f,1000.0f));
        }

        // get avoidance adjusted climb rate
        target_climb_rate = get_avoidance_adjusted_climbrate(target_climb_rate);

        // set position controller targets adjusted for pilot input
        takeoff.do_pilot_takeoff(target_climb_rate);

        // run loiter controller
        loiter_nav->update();

        // call attitude controller
        attitude_control->input_thrust_vector_rate_heading(loiter_nav->get_thrust_vector(), target_yaw_rate, false);
        break;

    case AltHoldModeState::Flying:
        // set motors to full range
        motors->set_desired_spool_state(AP_Motors::DesiredSpoolState::THROTTLE_UNLIMITED);

#if AC_PRECLAND_ENABLED
        bool precision_loiter_old_state = _precision_loiter_active;
        if (do_precision_loiter()) {
            precision_loiter_xy();
            _precision_loiter_active = true;
        } else {
            _precision_loiter_active = false;
        }
        if (precision_loiter_old_state && !_precision_loiter_active) {
            // prec loiter was active, not any more, let's init again as user takes control
            loiter_nav->init_target();
        }
        // run loiter controller if we are not doing prec loiter
        if (!_precision_loiter_active) {
            loiter_nav->update();
        }
#else
        loiter_nav->update();
#endif

        // get avoidance adjusted climb rate
        target_climb_rate = get_avoidance_adjusted_climbrate(target_climb_rate);

#if AP_RANGEFINDER_ENABLED
        // update the vertical offset based on the surface measurement
        copter.surface_tracking.update_surface_offset();
#endif

#if AC_PRECLAND_ENABLED
        // When precision loiter is active and pilot input is neutral, allow SET_POSITION_TARGET_LOCAL_NED to control yaw and altitude only
        bool use_external_yaw_z = false;
        if (_precision_loiter_active &&
            (AP_HAL::millis() - _external_target_ms) < _external_target_timeout_ms &&
            fabsf(target_climb_rate) < 2.0f &&
            fabsf(target_yaw_rate) < 2.0f) {
            use_external_yaw_z = true;
        }
        if (use_external_yaw_z) {
            // pos_control->set_pos_desired_z_cm(_external_z_cm);
            pos_control->set_pos_target_z_from_climb_rate_cm(_external_z_cm);
            // GCS_MAVLINK::send_text(MAV_SEVERITY_INFO, "Precision loiter active");
            // Use same angle-to-rate gain as guided mode (ANG_YAW_P, default 4.5): rate_cds = kP * error_cd
            const float yaw_error_cd = wrap_180_cd(_external_yaw_cd - ahrs.yaw_sensor);
            target_yaw_rate = yaw_error_cd * attitude_control->get_angle_yaw_p().kP();
            target_yaw_rate = constrain_float(target_yaw_rate, -yaw_rate_max_cds, yaw_rate_max_cds);
        } else {
            pos_control->set_pos_target_z_from_climb_rate_cm(target_climb_rate);
        }
#else
        // Send the commanded climb rate to the position controller
        pos_control->set_pos_target_z_from_climb_rate_cm(target_climb_rate);
#endif

        // call attitude controller (uses target_yaw_rate, possibly overridden by MAVLink when precision loiter + neutral)
        attitude_control->input_thrust_vector_rate_heading(loiter_nav->get_thrust_vector(), target_yaw_rate, false);
        break;
    }

    // run the vertical position controller and set output throttle
    pos_control->update_z_controller();
}

uint32_t ModeLoiter::wp_distance() const
{
    return loiter_nav->get_distance_to_target();
}

int32_t ModeLoiter::wp_bearing() const
{
    return loiter_nav->get_bearing_to_target();
}

#endif
