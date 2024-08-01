# Roboticd_pd
- Changed version from `Robotics`, based on `Maniskill3.BaseEnv`.
- Aiming at combine powerful framework `Maniskill3` and  `ros modules` for real alignment in only one structure.

## Details:
- `pdsimulator.py`: define PDSimulator to setup Maniskill3 (with Agent and basic environment like ground,etc.)
- `pdcloth_env.py`: define PDclothEnv to add pdsystem and simulate the deformable objects 
    - now add xlab_scene in it
- `tester/test_pd_panda_cloth.py`: test file to test the panda robot and deformable cloth in xlab_scene 
    - haven't add pdcomponent to every actors yet, but soon willed be added
- `environs/`: temporarily revise `sapien_square_wall.py` and `sapien_wall.py` (change into `world: PDSimulator`), other environments aren't revised yet(still `world: Simulator`) 
- `engines/`: will not be used anymore, the simulation is based on `Maniskill3.BaseEnv`
- `assets/`: some objects
- `examples/`and `prototypes`: original folders of sapien_pd 
- `robot/`, `sensor/` and `ros_plugins`: origin folders of Robotics
- `entity.py`: will not be used anymore (maybe)
- `cloth_env.py`, `simulator.py` and `simulator_base.py`: origin version of `pdcloth_env.py` and `pdsimulator.py`(combined `simulator.py` and `simulator_base.py`)

## Todo
- support `Mobile_sapien` robot as `Maniskill3.Agent`
- solve the `ros_plugins` compability problem

