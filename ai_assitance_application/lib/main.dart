import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'dart:typed_data';
import 'package:image/image.dart' as img;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final backCamera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.back,
    orElse: () => cameras.first,
  );
  runApp(MyApp(camera: backCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({Key? key, required this.camera}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: AIAssistant(camera: camera),
    );
  }
}

class AIAssistant extends StatefulWidget {
  final CameraDescription camera;

  const AIAssistant({Key? key, required this.camera}) : super(key: key);

  @override
  _AIAssistantState createState() => _AIAssistantState();
}

class _AIAssistantState extends State<AIAssistant> {
  late CameraController _cameraController;
  late stt.SpeechToText _speechToText;
  late FlutterTts _flutterTts;
  bool _isListening = false;
  String _debugText = '';
  bool _isCameraReady = false;
  bool _isProcessing = false;
  Uint8List? _imageBytes;
  Timer? _debounceTimer;
  String _currentInput = '';

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _speechToText = stt.SpeechToText();
    _flutterTts = FlutterTts();
  }

  Future<void> _initializeCamera() async {
    _cameraController =
        CameraController(widget.camera, ResolutionPreset.medium);
    try {
      await _cameraController.initialize();
      setState(() {
        _isCameraReady = true;
      });
    } catch (e) {
      setState(() {
        _debugText = 'Failed to initialize camera: $e';
      });
    }
  }

  Future<void> _processInput() async {
    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
    });

    if (!_isListening) {
      bool available = await _speechToText.initialize();
      if (available) {
        setState(() {
          _isListening = true;
          _debugText = 'Listening...';
        });
        _speechToText.listen(
          onResult: (result) => _processVoiceInput(result.recognizedWords),
        );
      } else {
        setState(() {
          _debugText = 'Speech recognition not available';
          _isProcessing = false;
        });
      }
    } else {
      setState(() {
        _isListening = false;
        _debugText = 'Stopped listening';
      });
      _speechToText.stop();
    }

    setState(() {
      _isProcessing = false;
    });
  }

  Future<void> _processVoiceInput(String text) async {
    _currentInput = text;

    if (_debounceTimer?.isActive ?? false) _debounceTimer!.cancel();

    _debounceTimer = Timer(Duration(milliseconds: 800), () {
      _sendInputToBackend(_currentInput);
    });
  }

  Future<void> _sendInputToBackend(String text) async {
    setState(() {
      _debugText = 'Processing: $text';
    });

    try {
      if (!_isCameraReady) {
        throw Exception('Camera is not ready');
      }

      final image = await _cameraController.takePicture();
      _imageBytes = await image.readAsBytes();

      String base64Image = await compressAndEncodeImage(_imageBytes!);

      setState(() {
        _debugText += '\nSending request to server...';
      });

      final response = await http
          .post(
            Uri.parse('http://10.199.23.88:8080/process'),
            headers: <String, String>{
              'Content-Type': 'application/json; charset=UTF-8',
            },
            body: jsonEncode(<String, dynamic>{
              'image': base64Image,
              'text': text,
              'session_id': 'default_session',
            }),
          )
          .timeout(Duration(seconds: 30));

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        String aiResponse = jsonResponse['response'];

        setState(() {
          _debugText += '\nReceived response: $aiResponse';
        });

        await _flutterTts.speak(aiResponse);
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      if (e is TimeoutException) {
        setState(() {
          _debugText += '\nRequest timed out. Please try again.';
        });
      } else {
        setState(() {
          _debugText += '\nAn error occurred: $e';
        });
      }
    } finally {
      setState(() {
        _isListening = false;
        _isProcessing = false;
      });
    }
  }

  Future<String> compressAndEncodeImage(Uint8List imageBytes) async {
    img.Image? image = img.decodeImage(imageBytes);
    if (image == null) return '';

    img.Image compressedImage = img.copyResize(image, width: 800);
    List<int> compressedBytes = img.encodeJpg(compressedImage, quality: 85);
    return base64Encode(compressedBytes);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // App title
          Column(
            children: [
              Container(
                color: Colors.blue,
                width: double.infinity,
                padding: EdgeInsets.only(
                    top: MediaQuery.of(context).padding.top + 10, bottom: 10),
                child: Text(
                  'My AI Assistant',
                  style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white),
                  textAlign: TextAlign.center,
                ),
              ),
              // Image area
              Expanded(
                child: Container(
                  color: Colors.white,
                  width: double.infinity,
                  child: _imageBytes != null
                      ? Image.memory(
                          _imageBytes!,
                          width: double.infinity, // Full width
                          height: double.infinity, // Full height
                          fit: BoxFit.fill, // Make it cover the container
                        )
                      : Center(child: Text('No image captured yet')),
                ),
              ),
              // Text response area (full width and height)
              if (_debugText.isNotEmpty)
                Container(
                  height: 160,
                  width: double.infinity,
                  color: Colors.grey[200],
                  child: SingleChildScrollView(
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(_debugText),
                    ),
                  ),
                ),
            ],
          ),
          // Floating Action Button with Microphone Icon
          Positioned(
            right: 20, // Position it towards the right
            bottom:
                170, // Position it above the text response area (adjustable)
            child: FloatingActionButton.large(
              onPressed: _isProcessing ? null : _processInput,
              child: Icon(
                _isListening
                    ? Icons.stop
                    : Icons.mic, // Change icon based on listening state
                size: 36, // Make the icon larger for visibility
              ),
              backgroundColor: Colors.blue, // Customize the button color
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _speechToText.cancel();
    _flutterTts.stop();
    _debounceTimer?.cancel();
    super.dispose();
  }
}


////////////older code 


// import 'dart:async';
// import 'dart:convert';
// import 'package:flutter/material.dart';
// import 'package:camera/camera.dart';
// import 'package:speech_to_text/speech_to_text.dart' as stt;
// import 'package:flutter_tts/flutter_tts.dart';
// import 'package:http/http.dart' as http;
// import 'dart:typed_data';
// import 'package:image/image.dart' as img;

// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   final cameras = await availableCameras();
//   final backCamera = cameras.firstWhere(
//     (camera) => camera.lensDirection == CameraLensDirection.back,
//     orElse: () => cameras.first,
//   );
//   runApp(MyApp(camera: backCamera));
// }

// class MyApp extends StatelessWidget {
//   final CameraDescription camera;

//   const MyApp({Key? key, required this.camera}) : super(key: key);

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       home: AIAssistant(camera: camera), 
//     );
//   }
// }

// class AIAssistant extends StatefulWidget {
//   final CameraDescription camera;

//   const AIAssistant({Key? key, required this.camera}) : super(key: key);

//   @override
//   _AIAssistantState createState() => _AIAssistantState();
// }

// class _AIAssistantState extends State<AIAssistant> {
//   late CameraController _cameraController;
//   late stt.SpeechToText _speechToText;
//   late FlutterTts _flutterTts;
//   bool _isListening = false;
//   String _debugText = '';
//   bool _isCameraReady = false;
//   bool _isProcessing = false;
//   Uint8List? _imageBytes;
//   Timer? _debounceTimer;
//   String _currentInput = '';

//   @override
//   void initState() {
//     super.initState();
//     _initializeCamera();
//     _speechToText = stt.SpeechToText();
//     _flutterTts = FlutterTts();
//   }

//   Future<void> _initializeCamera() async {
//     _cameraController = CameraController(widget.camera, ResolutionPreset.medium);
//     try {
//       await _cameraController.initialize();
//       setState(() {
//         _isCameraReady = true;
//       });
//     } catch (e) {
//       setState(() {
//         _debugText = 'Failed to initialize camera: $e';
//       });
//     }
//   }

//   Future<void> _processInput() async {
//     if (_isProcessing) return;

//     setState(() {
//       _isProcessing = true;
//     });

//     if (!_isListening) {
//       bool available = await _speechToText.initialize();
//       if (available) {
//         setState(() {
//           _isListening = true;
//           _debugText = 'Listening...';
//         });
//         _speechToText.listen(
//           onResult: (result) => _processVoiceInput(result.recognizedWords),
//         );
//       } else {
//         setState(() {
//           _debugText = 'Speech recognition not available';
//           _isProcessing = false;
//         });
//       }
//     } else {
//       setState(() {
//         _isListening = false;
//         _debugText = 'Stopped listening';
//       });
//       _speechToText.stop();
//     }

//     setState(() {
//       _isProcessing = false;
//     });
//   }

//   Future<void> _processVoiceInput(String text) async {
//     _currentInput = text;
    
//     if (_debounceTimer?.isActive ?? false) _debounceTimer!.cancel();
    
//     _debounceTimer = Timer(Duration(milliseconds: 600), () {
//       _sendInputToBackend(_currentInput);
//     });
//   }

//   Future<void> _sendInputToBackend(String text) async {
//     setState(() {
//       _debugText = 'Processing: $text';
//     });

//     try {
//       if (!_isCameraReady) {
//         throw Exception('Camera is not ready');
//       }

//       final image = await _cameraController.takePicture();
//       _imageBytes = await image.readAsBytes();

//       String base64Image = await compressAndEncodeImage(_imageBytes!);

//       setState(() {
//         _debugText += '\nSending request to server...';
//       });

//       final response = await http.post(
//         Uri.parse('http://10.199.20.206:8080/process'),
//         headers: <String, String>{
//           'Content-Type': 'application/json; charset=UTF-8',
//         },
//         body: jsonEncode(<String, dynamic>{
//           'image': base64Image,
//           'text': text,
//           'session_id': 'default_session',
//         }),
//       ).timeout(Duration(seconds: 30));

//       if (response.statusCode == 200) {
//         final jsonResponse = jsonDecode(response.body);
//         String aiResponse = jsonResponse['response'];

//         setState(() {
//           _debugText += '\nReceived response: $aiResponse';
//         });

//         await _flutterTts.speak(aiResponse);
//       } else {
//         throw Exception('Server error: ${response.statusCode}');
//       }
//     } catch (e) {
//       if (e is TimeoutException) {
//         setState(() {
//           _debugText += '\nRequest timed out. Please try again.';
//         });
//       } else {
//         setState(() {
//           _debugText += '\nAn error occurred: $e';
//         });
//       }
//       // Implement retry logic if needed
//     } finally {
//       setState(() {
//         _isListening = false;
//         _isProcessing = false;
//       });
//     }
//   }

//   Future<String> compressAndEncodeImage(Uint8List imageBytes) async {
//     img.Image? image = img.decodeImage(imageBytes);
//     if (image == null) return '';
    
//     img.Image compressedImage = img.copyResize(image, width: 800);
//     List<int> compressedBytes = img.encodeJpg(compressedImage, quality: 85);
//     return base64Encode(compressedBytes);
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('Blind Assistant App')),
//       body: Padding(
//         padding: const EdgeInsets.all(20.0),
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             ElevatedButton(
//               onPressed: _isProcessing ? null : _processInput,
//               child: Text(_isListening ? 'Stop Listening' : 'Start Listening'),
//             ),
//             if (_imageBytes != null)
//               Image.memory(
//                 _imageBytes!,
//                 width: 200,
//                 height: 200,
//               ),
//             SizedBox(height: 20),
//             Expanded(
//               child: SingleChildScrollView(
//                 child: Text(_debugText),
//               ),
//             ),
//           ],
//         ),
//       ),
//     );
//   }

//   @override
//   void dispose() {
//     _cameraController.dispose();
//     _speechToText.cancel();
//     _flutterTts.stop();
//     _debounceTimer?.cancel();
//     super.dispose();
//   }
// }