#pragma once

/// @file    AC_AttitudeControl_Tiltrotor.h
/// @brief   ArduCopter attitude control library

#include "AC_AttitudeControl.h"
#include "AC_AttitudeControl_Multi.h"
#include <AP_Motors/AP_MotorsMulticopter.h>


class AC_AttitudeControl_Tiltrotor : public AC_AttitudeControl_Multi {
public:
	AC_AttitudeControl_Tiltrotor(AP_AHRS_View &ahrs, const AP_MultiCopter &aparm, AP_MotorsMulticopter& motors);

    // user settable parameters
    static const struct AP_Param::GroupInfo var_info[];

protected:
};
