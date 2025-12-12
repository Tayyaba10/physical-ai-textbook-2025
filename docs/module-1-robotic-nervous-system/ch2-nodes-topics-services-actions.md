---
title: Ch2 - Nodes, Topics, Services & Actions
module: 1
chapter: 2
sidebar_label: Ch2: Nodes, Topics, Services & Actions
description: Understanding the fundamental communication patterns in ROS 2
tags: [ros2, nodes, topics, services, actions, communication]
difficulty: beginner
estimated_duration: 60
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Nodes, Topics, Services & Actions

## Learning Outcomes
- Understand the fundamental communication patterns in ROS 2
- Distinguish between topics, services, and actions
- Implement publishers and subscribers for topic-based communication
- Implement clients and servers for service-based communication
- Implement action clients and servers for goal-oriented communication
- Choose the appropriate communication pattern for different use cases

## Theory

### Nodes in ROS 2

In ROS 2, a **node** is an executable process that participates in the ROS computation. Nodes are the fundamental building blocks of ROS 2 applications. Each node typically performs a specific task and communicates with other nodes through topics, services, or actions.

Key characteristics of nodes:
- Each node has a unique name
- Nodes can be written in different programming languages (C++, Python, etc.)
- Nodes can be run independently
- Nodes contain publishers, subscribers, clients, and services

### Topic-Based Communication

<MermaidDiagram chart={`
graph LR;
    A[Publisher Node] --> B[Topic];
    C[Subscriber Node] --> B;
    D[Subscriber Node] --> B;
    B --> E[Message];
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style C fill:#2196F3,stroke:#0D47A1,color:#fff;
    style D fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

**Topics** enable asynchronous, decoupled communication between nodes. The communication is many-to-many - multiple publishers can send messages to a topic, and multiple subscribers can receive messages from the same topic.

Characteristics:
- **Asynchronous**: Publishers and subscribers don't need to be active at the same time
- **Fire-and-forget**: Publishers send messages without expecting responses
- **Broadcast**: One message can be received by multiple subscribers
- **Unidirectional**: Data flows in one direction (publisher → topic → subscribers)

### Service-Based Communication

<MermaidDiagram chart={`
graph LR;
    A[Client] --> B[Request];
    B --> C[Service Server];
    C --> D[Response];
    D --> A;
    style A fill:#FF9800,stroke:#E65100,color:#fff;
    style C fill:#9C27B0,stroke:#4A148C,color:#fff;
`} />

**Services** enable synchronous request-response communication between nodes. When a client sends a request to a service, it waits for a response before continuing.

Characteristics:
- **Synchronous**: Client waits for response
- **Request-response pattern**: One request gets one response
- **Stateless**: Each request is independent
- **Suitable for**: Tasks that have clear start and end points

### Action-Based Communication

<MermaidDiagram chart={`
graph TD;
    A[Action Client] --> B[Send Goal];
    B --> C[Action Server];
    C --> D[Feedback];
    D --> A;
    C --> E[Result];
    E --> A;
    style A fill:#00BCD4,stroke:#006064,color:#fff;
    style C fill:#E91E63,stroke:#880E4F,color:#fff;
`} />

**Actions** are used for long-running tasks that provide feedback during execution and ultimately return a result. They're ideal for tasks like navigation, where you want to track progress.

Characteristics:
- **Long-running**: Suitable for operations that take significant time
- **Feedback**: Provides ongoing feedback during execution
- **Goal-oriented**: Supports canceling, pausing, and preemption
- **Two-way**: Combines request-response with continuous feedback

## Step-by-Step Labs

### Lab 1: Creating a Publisher and Subscriber

1. **Create a new package** for your communication examples:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_cmake cpp_pubsub
   cd cpp_pubsub
   ```

2. **Create a publisher source file** (`src/talker.cpp`):
   ```cpp
   #include <chrono>
   #include <functional>
   #include <memory>
   #include <string>
   
   #include "rclcpp/rclcpp.hpp"
   #include "std_msgs/msg/string.hpp"
   
   using namespace std::chrono_literals;
   
   class MinimalPublisher : public rclcpp::Node
   {
   public:
     MinimalPublisher()
     : Node("minimal_publisher"), count_(0)
     {
       publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
       timer_ = this->create_wall_timer(
         500ms, std::bind(&MinimalPublisher::timer_callback, this));
     }
   
   private:
     void timer_callback()
     {
       auto message = std_msgs::msg::String();
       message.data = "Hello, world! " + std::to_string(count_++);
       RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
       publisher_->publish(message);
     }
     rclcpp::TimerBase::SharedPtr timer_;
     rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
     size_t count_;
   };
   
   int main(int argc, char * argv[])
   {
     rclcpp::init(argc, argv);
     rclcpp::spin(std::make_shared<MinimalPublisher>());
     rclcpp::shutdown();
     return 0;
   }
   ```

3. **Create a subscriber source file** (`src/listener.cpp`):
   ```cpp
   #include <memory>
   
   #include "rclcpp/rclcpp.hpp"
   #include "std_msgs/msg/string.hpp"
   
   class MinimalSubscriber : public rclcpp::Node
   {
   public:
     MinimalSubscriber()
     : Node("minimal_subscriber")
     {
       subscription_ = this->create_subscription<std_msgs::msg::String>(
         "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1));
     }
   
   private:
     void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
     {
       RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
     }
     rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
   };
   
   int main(int argc, char * argv[])
   {
     rclcpp::init(argc, argv);
     rclcpp::spin(std::make_shared<MinimalSubscriber>());
     rclcpp::shutdown();
     return 0;
   }
   ```

4. **Update CMakeLists.txt**:
   ```cmake
   cmake_minimum_required(VERSION 3.8)
   project(cpp_pubsub)
   
   if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     add_compile_options(-Wall -Wextra -Wpedantic)
   endif()
   
   find_package(ament_cmake REQUIRED)
   find_package(rclcpp REQUIRED)
   find_package(std_msgs REQUIRED)
   
   add_executable(talker src/talker.cpp)
   add_executable(listener src/listener.cpp)
   
   ament_target_dependencies(talker rclcpp std_msgs)
   ament_target_dependencies(listener rclcpp std_msgs)
   
   install(TARGETS
     talker
     listener
     DESTINATION lib/${PROJECT_NAME})
   
   ament_package()
   ```

### Lab 2: Creating a Service Client and Server

1. **Create a service definition** (`srv/AddTwoInts.srv`):
   ```srv
   int64 a
   int64 b
   ---
   int64 sum
   ```

2. **Create a service server** (`src/add_two_ints_server.cpp`):
   ```cpp
   #include "rclcpp/rclcpp.hpp"
   #include "example_interfaces/srv/add_two_ints.hpp"
   
   class MinimalService : public rclcpp::Node
   {
   public:
     MinimalService()
     : Node("minimal_service")
     {
       service_ = this->create_service<example_interfaces::srv::AddTwoInts>(
         "add_two_ints",
         std::bind(&MinimalService::add, this, std::placeholders::_1, std::placeholders::_2));
     }
   
   private:
     void add(const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
             const example_interfaces::srv::AddTwoInts::Response::SharedPtr response)
     {
       response->sum = request->a + request->b;
       RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "%ld + %ld = %ld",
                   request->a, request->b, response->sum);
     }
     rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
   };
   
   int main(int argc, char * argv[])
   {
     rclcpp::init(argc, argv);
     rclcpp::spin(std::make_shared<MinimalService>());
     rclcpp::shutdown();
     return 0;
   }
   ```

3. **Create a service client** (`src/add_two_ints_client.cpp`):
   ```cpp
   #include "rclcpp/rclcpp.hpp"
   #include "example_interfaces/srv/add_two_ints.hpp"
   
   #include <chrono>
   #include <cstdlib>
   #include <memory>
   
   using namespace std::chrono_literals;
   
   int main(int argc, char * argv[])
   {
     rclcpp::init(argc, argv);
   
     if (argc != 3) {
         RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "usage: add_two_ints_client X Y");
         return 1;
     }
   
     std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("add_two_ints_client");
     rclcpp::Client<example_interfaces::srv::AddTwoInts>::SharedPtr client =
       node->create_client<example_interfaces::srv::AddTwoInts>("add_two_ints");
   
     while (!client->wait_for_service(1s)) {
       if (!rclcpp::ok()) {
         RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
         return 0;
       }
       RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Service not available, waiting again...");
     }
   
     auto request = std::make_shared<example_interfaces::srv::AddTwoInts::Request>();
     request->a = atoll(argv[1]);
     request->b = atoll(argv[2]);
   
     auto result = client->async_send_request(request);
     if (rclcpp::spin_until_future_complete(node, result) ==
       rclcpp::FutureReturnCode::SUCCESS)
     {
       RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "%ld + %ld = %ld",
                   request->a, request->b, result.get()->sum);
     } else {
       RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to call service add_two_ints");
     }
   
     rclcpp::shutdown();
     return 0;
   }
   ```

### Lab 3: Creating an Action Client and Server

1. **Create an action server** (`src/fibonacci_action_server.cpp`):
   ```cpp
   #include <functional>
   #include <memory>
   #include <thread>
   
   #include "rclcpp/rclcpp.hpp"
   #include "rclcpp_action/rclcpp_action.hpp"
   #include "example_interfaces/action/fibonacci.hpp"
   
   class FibonacciActionServer : public rclcpp::Node
   {
   public:
     using Fibonacci = example_interfaces::action::Fibonacci;
     using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;
   
     FibonacciActionServer() : Node("fibonacci_action_server")
     {
       using namespace std::placeholders;
   
       action_server_ = rclcpp_action::create_server<Fibonacci>(
         this,
         "fibonacci",
         std::bind(&FibonacciActionServer::handle_goal, this, _1, _2),
         std::bind(&FibonacciActionServer::handle_cancel, this, _1),
         std::bind(&FibonacciActionServer::handle_accepted, this, _1));
     }
   
   protected:
     rclcpp_action::GoalResponse handle_goal(
       const rclcpp_action::GoalUUID & uuid,
       std::shared_ptr<const Fibonacci::Goal> goal)
     {
       RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
       (void)uuid;
       return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
     }
   
     rclcpp_action::CancelResponse handle_cancel(
       const std::shared_ptr<GoalHandleFibonacci> goal_handle)
     {
       RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
       (void)goal_handle;
       return rclcpp_action::CancelResponse::ACCEPT;
     }
   
     void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
     {
       using namespace std::placeholders;
       std::thread{std::bind(&FibonacciActionServer::execute, this, _1), goal_handle}.detach();
     }
   
   private:
     rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;
   
     void execute(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
     {
       RCLCPP_INFO(this->get_logger(), "Executing goal");
   
       rclcpp::Rate loop_rate(1);
       const auto goal = goal_handle->get_goal();
       auto feedback = std::make_shared<Fibonacci::Feedback>();
       auto result = std::make_shared<Fibonacci::Result>();
   
       std::vector<int> sequence = {0, 1};
       feedback->sequence = sequence;
       result->sequence = sequence;
   
       for (int i = 1; (i < goal->order) && rclcpp::ok(); ++i) {
         if (goal_handle->is_canceling()) {
           RCLCPP_INFO(this->get_logger(), "Goal canceled");
           result->sequence = sequence;
           goal_handle->canceled(result);
           RCLCPP_INFO(this->get_logger(), "Sending canceled state");
           return;
         }
   
         sequence.push_back(sequence[i] + sequence[i - 1]);
         feedback->sequence = sequence;
         goal_handle->publish_feedback(feedback);
         RCLCPP_INFO(this->get_logger(), "Publishing feedback");
   
         loop_rate.sleep();
       }
   
       if (rclcpp::ok()) {
         result->sequence = sequence;
         goal_handle->succeed(result);
         RCLCPP_INFO(this->get_logger(), "Goal succeeded");
       }
     }
   };
   
   int main(int argc, char ** argv)
   {
     rclcpp::init(argc, argv);
     auto action_server = std::make_shared<FibonacciActionServer>();
     rclcpp::spin(action_server);
     rclcpp::shutdown();
     return 0;
   }
   ```

## Runnable Code Example

Here's a Python example showing all three communication patterns combined:

```python
# combined_communication_example.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts
from example_interfaces.action import Fibonacci
import rclpy.action


class CommunicationNode(Node):
    def __init__(self):
        super().__init__('communication_node')
        
        # Topic publisher
        self.publisher = self.create_publisher(String, 'combined_topic', 10)
        
        # Service client
        self.cli = self.create_client(AddTwoInts, 'add_two_ints_client')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        # Action client
        self._action_client = rclpy.action.ActionClient(
            self,
            Fibonacci,
            'fibonacci_action_client'
        )

        # Topic subscriber
        self.subscription = self.create_subscription(
            String,
            'combined_topic',
            self.topic_callback,
            10)
        
        # Timer to trigger all communications
        self.timer = self.create_timer(5.0, self.timer_callback)
    
    def topic_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
    
    def timer_callback(self):
        # Publish to topic
        msg = String()
        msg.data = f'Hello, combined communication! Time: {self.get_clock().now().seconds_nanoseconds()}'
        self.publisher.publish(msg)
        
        # Call service
        self.call_service()
        
        # Send action goal
        self.send_action_goal()
    
    def call_service(self):
        request = AddTwoInts.Request()
        request.a = 2
        request.b = 3
        future = self.cli.call_async(request)
        future.add_done_callback(self.service_callback)
    
    def service_callback(self, future):
        result = future.result()
        self.get_logger().info(f'Result of service call: {result.sum}')
    
    def send_action_goal(self):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = 10
        
        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        
        self.get_logger().info('Goal accepted')
        
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')
    
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Final result: {result.sequence}')


def main(args=None):
    rclpy.init(args=args)
    communication_node = CommunicationNode()
    
    try:
        rclpy.spin(communication_node)
    except KeyboardInterrupt:
        pass
    finally:
        communication_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Mini-project

Create a ROS 2 package that demonstrates all three communication patterns:

1. A publisher that publishes sensor data (temperature, humidity) at regular intervals
2. A service server that takes a sensor reading and returns whether the reading is within normal range
3. An action server that simulates a complex task like "moving_to_location" with feedback on progress

Create a node that:
- Publishes sensor data to the topic
- Calls the service to check if readings are normal
- Sends a goal to an action server to simulate robot movement
- Subscribes to a topic to receive status updates

## Summary

This chapter covered the three fundamental communication patterns in ROS 2:

- **Topics**: For asynchronous, decoupled communication using publisher-subscriber model
- **Services**: For synchronous request-response communication
- **Actions**: For long-running tasks with feedback and goals

Each communication pattern has its use cases:
- Use topics for streaming data or broadcasting information
- Use services for tasks with clear inputs and outputs
- Use actions for long-running tasks that provide feedback during execution

Understanding these patterns is crucial for designing well-architected ROS 2 systems.