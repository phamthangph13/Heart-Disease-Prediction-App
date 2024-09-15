import 'package:flutter/material.dart';
import 'package:project/CameraScreen.dart';
import 'package:project/ChatBot.dart';
import 'package:project/Home.dart';

class MainScreen extends StatefulWidget {
  @override
  _MainScreenState createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  final List<Widget> _pages = [
    HomeScreen(),
    CameraScreen(),
    ChatBotScreen(), // Replace with your actual screen
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_currentIndex], // Display the selected screen
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: [
          BottomNavigationBarItem(
            icon: Icon(Icons.input_outlined,size: 30,),
            label: '', // Label is left empty
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_front,size: 30),
            label: '', // Label is left empty
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.insert_chart,size: 30),
            label: '', // Label is left empty
          ),
        ],
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.grey,
        showUnselectedLabels: false, // Hide labels when unselected
        type: BottomNavigationBarType.fixed,
      ),
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: MainScreen(),
    debugShowCheckedModeBanner: false, // Hide the debug banner
  ));
}
