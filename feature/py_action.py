import numpy as np
from pysc2.lib import actions
import common.utils as U
from pysc2.lib import actions, features

## General Player Information ##

FUNCTIONS = actions.FUNCTIONS


## Action Space ##

## Action ID [3]
## Point / Unit Action [4]
## Unit ID [500]
## Point [84,84]
## select_add [2]

class ActionTransform:

    def transform(self, action_plan):

        if action_plan['action_id'] == 0:
            action_plan = [FUNCTIONS.select_unit.id, [action_plan['action_act']], [action_plan['unit_id']]]
        elif action_plan['action_id'] == 1:
            action_plan = [FUNCTIONS.select_point.id, [action_plan['action_act']],  action_plan['screen_point']]
        elif action_plan['action_id'] == 2:
            action_plan = [FUNCTIONS.select_army.id, [action_plan['select_add']]]
        else:
            action_plan = [FUNCTIONS.no_op.id]

        return action_plan
