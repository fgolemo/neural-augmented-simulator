import time

from poppy_helpers.constants import MOVES
from poppy_helpers.controller import SwordFightController
from pypot.robot import from_remote

poppy_def = from_remote('flogo4.local', 4242)

controller_def = SwordFightController(poppy_def, mode="def")
controller_def.rest()

controller_def.compliant(False)
controller_def.set_max_speed(150)

#
# while True:
#     print (controller_def.get_pos())
#     time.sleep(2)
#
#
# # [-14.52, -20.09, 65.54, 98.97, -21.85, -53.81]
# # [8.36, -27.13, 65.84, -75.51, -62.32, -25.37]
# # [-39.15, -10.7, 101.91, 96.92, 20.97, -70.53]
# # [-13.64, 2.49, 41.79, 82.55, -85.19, 27.71]

for i in range(5):
    controller_def.goto_pos(MOVES["def{}".format(i)])
    time.sleep(2)

controller_def.rest()