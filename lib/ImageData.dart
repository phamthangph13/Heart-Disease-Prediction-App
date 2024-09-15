import 'package:flutter/material.dart';
import 'dart:io' show File;
import 'dart:typed_data'; // For Uint8List
import 'package:image_picker/image_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import 'dart:convert'; // For JSON encoding/decoding
import 'package:dio/dio.dart'; // Import dio package for web support
import 'Details.dart'; // Import DetailsScreen

class ConfirmDataScreen extends StatefulWidget {
  final XFile image;

  ConfirmDataScreen({required this.image});

  @override
  _ConfirmDataScreenState createState() => _ConfirmDataScreenState();
}

class _ConfirmDataScreenState extends State<ConfirmDataScreen> {
  bool _isLoading = false;

  Future<void> _handleConfirmation() async {
    const String extractFromImageUrl = 'http://127.0.0.1:5000/extract_from_image';
    const String predictUrl = 'http://127.0.0.1:5000/predict';
    const String plotAccuracyUrl = 'http://127.0.0.1:5000/plot_model_accuracy';
    const String plotTargetUrl = 'http://127.0.0.1:5000/plot_target_distribution';
    const String plotDataUrl = 'http://127.0.0.1:5000/plot_data_distribution';
    const String plotChartUrl = 'http://127.0.0.1:5000/plot_chart';

    setState(() {
      _isLoading = true;
    });

    try {
      Map<String, String> headers = {
        'Content-Type': 'application/json',
      };

      Uint8List imageBytes;

      if (kIsWeb) {
        imageBytes = await widget.image.readAsBytes();
      } else {
        imageBytes = await File(widget.image.path).readAsBytes();
      }

      var request = http.MultipartRequest('POST', Uri.parse(extractFromImageUrl))
        ..files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: widget.image.name));

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var extractedData = json.decode(responseData) as Map<String, dynamic>;

      print('Extracted Data: $extractedData');

      // Check if the extracted data is empty
      if (extractedData.isEmpty) {
        throw Exception('No data extracted from the image.');
      }

      // Add default values for missing fields
      extractedData.putIfAbsent('age', () => 50); // Default age
      extractedData.putIfAbsent('sex', () => 1);  // Default sex: 1 = Male
      extractedData.putIfAbsent('cp', () => 0);   // Default chest pain type
      extractedData.putIfAbsent('trestbps', () => 120);  // Default resting blood pressure
      extractedData.putIfAbsent('chol', () => 200);  // Default cholesterol
      extractedData.putIfAbsent('fbs', () => 0);     // Default fasting blood sugar
      extractedData.putIfAbsent('restecg', () => 0); // Default resting electrocardiographic results
      extractedData.putIfAbsent('thalach', () => 150); // Default maximum heart rate
      extractedData.putIfAbsent('exang', () => 0);   // Default exercise induced angina
      extractedData.putIfAbsent('oldpeak', () => 0.0); // Default oldpeak value
      extractedData.putIfAbsent('slope', () => 1);   // Default slope
      extractedData.putIfAbsent('ca', () => 0);      // Default number of major vessels
      extractedData.putIfAbsent('thal', () => 2);    // Default thalassemia type

      // Make prediction
      final predictionResponse = await http.post(
        Uri.parse(predictUrl),
        headers: headers,
        body: json.encode(extractedData),
      );

      // Print the full response body for debugging
      final predictionResponseBody = predictionResponse.body;
      print('Prediction API Response: $predictionResponseBody');

      // Parse the prediction result
      final predictionResult = json.decode(predictionResponseBody)['prediction'] ?? 'No prediction available';

      // Fetch plots
      final fetchPlot = (String url) async {
        final plotResponse = await http.get(Uri.parse(url));
        return plotResponse.bodyBytes;
      };

      final modelAccuracyPlot = await fetchPlot(plotAccuracyUrl);
      final targetDistributionPlot = await fetchPlot(plotTargetUrl);
      final dataDistributionPlot = await fetchPlot(plotDataUrl);
      final chartPlot = await fetchPlot(plotChartUrl);

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => DetailsScreen(
            predictionResult: predictionResult,
            modelAccuracyPlot: modelAccuracyPlot,
            targetDistributionPlot: targetDistributionPlot,
            dataDistributionPlot: dataDistributionPlot,
            chartPlot: chartPlot,
          ),
        ),
      );
    } catch (error) {
      print('Error: $error');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to process the image.'),backgroundColor: Colors.redAccent,),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final isPortrait = MediaQuery.of(context).orientation == Orientation.portrait;
    final aspectRatio = isPortrait ? 9 / 16 : 16 / 9;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlueAccent,
        title: Text('Data Verification from Images',style: TextStyle(color: Colors.white),),
      ),
      body: Center(
        child: _isLoading
            ? CircularProgressIndicator(backgroundColor: Colors.lightBlueAccent,color: Colors.redAccent,)
            : Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Expanded(
              child: AspectRatio(
                aspectRatio: aspectRatio,
                child: kIsWeb
                    ? FutureBuilder<Uint8List>(
                  future: widget.image.readAsBytes(),
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.done) {
                      if (snapshot.hasData) {
                        return Image.memory(
                          snapshot.data!,
                          fit: BoxFit.cover,
                        );
                      } else {
                        return Center(child: Text('Error uploading the image! Please try again.'));
                      }
                    } else {
                      return Center(child: CircularProgressIndicator(backgroundColor: Colors.lightBlueAccent,color: Colors.redAccent,));
                    }
                  },
                )
                    : Image.file(
                  File(widget.image.path),
                  fit: BoxFit.cover,
                ),
              ),
            ),
            SizedBox(height: 10),
            ElevatedButton(
              style: ButtonStyle(
                backgroundColor: MaterialStateProperty.all<Color>(Colors.lightBlueAccent),
                padding: MaterialStateProperty.all<EdgeInsets>(EdgeInsets.symmetric(vertical: 12, horizontal: 20)),
                shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                  RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8.0),
                  ),
                ),
              ),
              onPressed: _handleConfirmation,
              child: Text(
                'Prediction',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
