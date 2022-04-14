## get joints
'''
curl http://127.0.0.1:7777/get_joint_positions | jq 
 
'''
## set joints
'''
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "names": ["panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"],
  "positions": [0,0.03,0,0,0,0,0]
}' http://127.0.0.1:7777/set_joint_positions | jq
'''