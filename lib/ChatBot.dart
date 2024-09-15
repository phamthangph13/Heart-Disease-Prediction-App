import 'package:flutter/material.dart';
import 'package:socket_io_client/socket_io_client.dart' as IO;

class ChatBotScreen extends StatefulWidget {
  @override
  _ChatBotScreenState createState() => _ChatBotScreenState();
}

class _ChatBotScreenState extends State<ChatBotScreen> {
  late IO.Socket socket;
  final TextEditingController _messageController = TextEditingController();
  final List<Map<String, String>> _messages = [];
  final ScrollController _scrollController = ScrollController();
  bool _isTyping = false; // Add this flag to indicate if bot is typing

  @override
  void initState() {
    super.initState();
    _connectToSocket();
  }

  void _connectToSocket() {
    socket = IO.io('http://127.0.0.1:5000', <String, dynamic>{
      'transports': ['websocket'],
    });

    socket.on('connect', (_) {
      print('Connected to WebSocket');
    });

    socket.on('response', (data) {
      final response = data['response'];
      if (response != null && response is String) {
        setState(() {
          _isTyping = false; // Bot is done typing
          _messages.add({'sender': 'Bot', 'message': response});
        });
        _scrollToBottom();
      } else {
        print('Received invalid response: $data');
      }
    });

    socket.on('typing', (_) {
      setState(() {
        _isTyping = true; // Bot is typing
      });
    });

    socket.on('disconnect', (_) {
      print('Disconnected from WebSocket');
    });
  }

  void _sendMessage() {
    final message = _messageController.text.trim();
    if (message.isNotEmpty) {
      setState(() {
        _messages.add({'sender': 'User', 'message': message});
      });
      socket.emit('message', {'message': message});
      _messageController.clear();
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    Future.delayed(Duration(milliseconds: 100), () {
      _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlueAccent,
        title: Text('Chat Bot',style: TextStyle(color: Colors.white),),
      ),
      body: Column(
        children: <Widget>[
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: _messages.length + (_isTyping ? 1 : 0),
              itemBuilder: (context, index) {
                if (index == _messages.length) {
                  return _buildTypingIndicator(); // Show typing indicator
                }
                final message = _messages[index];
                return ListTile(
                  title: Align(
                    alignment: message['sender'] == 'User'
                        ? Alignment.centerRight
                        : Alignment.centerLeft,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: <Widget>[
                        if (message['sender'] == 'Bot')
                          CircleAvatar(
                            backgroundColor: Colors.lightBlueAccent,
                            child: Icon(Icons.compass_calibration, color: Colors.white),
                          ),
                        SizedBox(width: 8),
                        Expanded(
                          child: Container(
                            padding: EdgeInsets.all(10),
                            decoration: BoxDecoration(
                              color: message['sender'] == 'User'
                                  ? Colors.blue
                                  : Colors.grey,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              message['message'] ?? '', // Handle null value
                              style: TextStyle(color: Colors.white),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: <Widget>[
                Expanded(
                  child: TextField(
                    controller: _messageController,
                    decoration: InputDecoration(
                      hintText: 'Type your message...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30.0), // Rounded corners
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30.0), // Rounded corners when focused
                        borderSide: BorderSide(color: Colors.blue, width: 2.0), // Customize border color and width
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30.0), // Rounded corners when not focused
                        borderSide: BorderSide(color: Colors.grey, width: 1.0), // Customize border color and width
                      ),
                    ),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.send),
                  onPressed: _sendMessage,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTypingIndicator() {
    return ListTile(
      title: Align(
        alignment: Alignment.centerLeft,
        child: Row(
          children: <Widget>[
            CircleAvatar(
              backgroundColor: Colors.grey,
              child: Icon(Icons.android, color: Colors.white),
            ),
            SizedBox(width: 8),
            Container(
              padding: EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.grey,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Container(
                    width: 8,
                    height: 8,
                    margin: EdgeInsets.symmetric(horizontal: 2),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white,
                    ),
                  ),
                  Container(
                    width: 8,
                    height: 8,
                    margin: EdgeInsets.symmetric(horizontal: 2),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white,
                    ),
                  ),
                  Container(
                    width: 8,
                    height: 8,
                    margin: EdgeInsets.symmetric(horizontal: 2),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    socket.dispose();
    _scrollController.dispose();
    super.dispose();
  }
}
