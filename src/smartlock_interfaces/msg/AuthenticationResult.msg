# Authentication result from brain node
# Contains decision to grant or deny access

builtin_interfaces/Time timestamp
sensor_msgs/Image image
bool authenticated       # True if authentication successful
string auth_type         # Type: "PIN", "RFID", "FACE"
float32 confidence       # recognition confidence (0.0 - 1.0)
