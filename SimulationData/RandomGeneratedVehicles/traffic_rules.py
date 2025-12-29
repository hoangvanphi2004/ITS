
"""
Traffic Signal Rule-Based Supervisor (Safety Layer)
Enforces constraints like Minimum Green Time to ensure safe and stable operation.
"""

class TrafficRules:
    def __init__(self, min_green_time=15, max_green_time=60):
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        
        # Track when the last phase change occurred for each junction
        # format: {junction_id: last_switch_time_step}
        self.last_switch_times = {}

    def is_switch_allowed(self, junction_id, current_step, current_phase_duration):
        """
        Check if switching phases is allowed based on Minimum Green Time.
        
        :param junction_id: ID of the traffic light
        :param current_step: Current simulation step
        :param current_phase_duration: How long the current phase has been active
        :return: (bool) True if switch is safe/allowed, False otherwise
        """
        
        # 1. Enforce Minimum Green Time
        # If the current phase has not run for at least min_green_time, block the switch.
        # Note: We assume yellow time is handled by SUMO's transition logic or a separate state.
        # Here we strictly check the active "Green" duration.
        
        if current_phase_duration < self.min_green_time:
            return False
            
        return True

    def get_safe_action(self, junction_id, proposed_action, current_phase, current_duration):
        """
        Filters the RL agent's action through safety rules.
        
        :param proposed_action: The phase index the RL wants to switch to.
        :param current_phase: The index of the current active phase.
        :param current_duration: How long current phase has run.
        :return: A safe action (either the Proposed Action or Override to Keep Current).
        """
        
        # If RL wants to keep same phase, always allow
        if proposed_action == current_phase:
            return proposed_action
            
        # If RL wants to switch
        if self.is_switch_allowed(junction_id, None, current_duration):
            return proposed_action
        else:
            # Rule Violation: Force agent to keep current phase
            # print(f"Safety Override: Triggered MinGreen constraint. Holding Phase {current_phase}.")
            return current_phase
