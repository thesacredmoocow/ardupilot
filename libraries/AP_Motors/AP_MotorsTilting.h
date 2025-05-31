#pragma once
#if AP_SCRIPTING_ENABLED

#include "AP_MotorsMatrix.h"

class AP_MotorsTilting : public AP_MotorsMatrix {
public:

    // Constructor
    AP_MotorsTilting(uint16_t speed_hz = AP_MOTORS_SPEED_DEFAULT) :
        AP_MotorsMatrix(speed_hz)
    {
        if (_singleton != nullptr) {
            AP_HAL::panic("AP_MotorsTilting must be singleton");
        }
        _singleton = this;
    };

    // get singleton instance
    static AP_MotorsTilting *get_singleton() {
        return _singleton;
    }

    struct factor_table {
        float roll[AP_MOTORS_MAX_NUM_MOTORS];
        float pitch[AP_MOTORS_MAX_NUM_MOTORS];
        float yaw[AP_MOTORS_MAX_NUM_MOTORS];
        float throttle[AP_MOTORS_MAX_NUM_MOTORS];
    };

    void init(motor_frame_class frame_class, motor_frame_type frame_type) override;

    // Init to be called from scripting
    bool init(uint8_t expected_num_motors) override {return true;};

    // add a interpolation point table
    void load_factors(const factor_table &table);

    // output - sends commands to the motors
    void output_to_motors() override;

    // set the target pitch angle positive is nose up
    void set_pitch_angle(float angle) override;

protected:

    void output_armed_stabilizing() override;

    // Do not apply thrust compensation, this is used by Quadplane tiltrotors
    // assume the compensation is done in the mixer and should not be done by quadplane
    void thrust_compensation(void) override {};

    const char* _get_frame_string() const override { return "Tiltrotor Quadcopter"; }

    float pitch_offset_angle;

private:
    static AP_MotorsTilting *_singleton;
};

#endif // AP_SCRIPTING_ENABLED
