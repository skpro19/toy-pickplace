| State          | Action                    | Transition condition           |
| -------------- | ------------------------- | ------------------------------ |
| `APPROACH`     | Move above cube           | EE is above cube               |
| `DESCEND`      | Move down to grasp height | EE is near cube                |
| `GRASP`        | Close gripper             | Gripper has closed for N steps |
| `LIFT`         | Move upward               | Cube is lifted                 |
| `MOVE_TO_TRAY` | Move above tray           | EE is above tray               |
| `LOWER`        | Move down into tray       | Cube is inside tray height     |
| `RELEASE`      | Open gripper              | Gripper opened for N steps     |
| `RETREAT`      | Move upward               | EE has moved away              |
| `DONE`         | Stop                      | Cube is stable in tray         |
