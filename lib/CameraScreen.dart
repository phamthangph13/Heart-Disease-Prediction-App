import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io' show File;
import 'dart:html' as html;
import 'dart:typed_data'; // For Uint8List
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';
import 'ImageData.dart';

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  XFile? _capturedImage;
  bool _isCameraInitialized = false;
  html.VideoElement? _webcamVideoElement;
  final ImagePicker _picker = ImagePicker(); // Initialize picker for selecting images

  @override
  void initState() {
    super.initState();
    if (kIsWeb) {
      _initializeWebCamera();
    } else {
      _initializeCamera();
    }
  }

  // Initialize camera for mobile platforms
  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    if (_cameras != null && _cameras!.isNotEmpty) {
      _controller = CameraController(
        _cameras![0],
        ResolutionPreset.high,
      );

      await _controller?.initialize();
      setState(() {
        _isCameraInitialized = true;
      });
    }
  }

  // Initialize camera for web platform
  Future<void> _initializeWebCamera() async {
    _webcamVideoElement = html.VideoElement();
    html.window.navigator.getUserMedia(video: true).then((stream) {
      _webcamVideoElement!.srcObject = stream;
      _webcamVideoElement!.autoplay = true;
      setState(() {
        _isCameraInitialized = true;
      });
    }).catchError((err) {
      print('Error accessing camera: $err');
    });
  }

  Future<void> _takePicture() async {
    if (kIsWeb) {
      // Web camera: display video from camera
      setState(() {
        _capturedImage = null; // You may store the video stream if needed
      });
    } else {
      if (_controller != null && _controller!.value.isInitialized) {
        final image = await _controller?.takePicture();
        setState(() {
          _capturedImage = image;
        });

        if (_capturedImage != null) {
          // Navigate to ConfirmDataScreen after capturing image
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ConfirmDataScreen(image: _capturedImage!),
            ),
          );
        }
      }
    }
  }

  // Select image from gallery
  Future<void> _pickImage() async {
    final pickedImage = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedImage != null) {
      setState(() {
        _capturedImage = pickedImage;
      });

      // Navigate to ConfirmDataScreen after picking image
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ConfirmDataScreen(image: _capturedImage!),
        ),
      );
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlueAccent,
        title: Text(
          'Heart Disease Prediction With Camera Scan',style: TextStyle(color: Colors.white),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: _isCameraInitialized
                ? kIsWeb
                ? HtmlElementView(viewType: 'webcamVideoElement')
                : CameraPreview(_controller!)
                : Center(child: CircularProgressIndicator()),
          ),
          SizedBox(height: 10),
          if (_capturedImage != null)
            kIsWeb
                ? FutureBuilder<Uint8List>(
              future: _capturedImage!.readAsBytes(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  if (snapshot.hasData) {
                    return Image.memory(
                      snapshot.data!,
                      width: 100,
                      height: 100,
                      fit: BoxFit.cover,
                    );
                  } else {
                    return Text('Error loading image');
                  }
                } else {
                  return CircularProgressIndicator();
                }
              },
            )
                : Image.file(
              File(_capturedImage!.path),
              width: 100,
              height: 100,
              fit: BoxFit.cover,
            ),
          SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Take picture button
              ElevatedButton(
                onPressed: _takePicture,
                child: Icon(Icons.camera_alt,color: Colors.white,),
                style: ButtonStyle(
                  backgroundColor: MaterialStateProperty.all(Colors.lightBlueAccent),
                ),
              ),
              SizedBox(width: 20), // Space between buttons
              // Select image button
              ElevatedButton(
                onPressed: _pickImage,
                child: Icon(Icons.photo_library,color: Colors.white,),
                style: ButtonStyle(
                  backgroundColor: MaterialStateProperty.all(Colors.lightBlueAccent),
                ),
              ),
            ],
          ),
          SizedBox(height: 20),
        ],
      ),
    );
  }
}
