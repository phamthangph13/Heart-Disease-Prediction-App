import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'Details.dart'; // Import the file containing the DetailsScreen
import 'package:project/tutorial.dart'; // Import the file containing the HelpSheet

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _formKey = GlobalKey<FormState>();
  final _ageController = TextEditingController();
  final _sexController = TextEditingController();
  final _cpController = TextEditingController();
  final _trestbpsController = TextEditingController();
  final _cholController = TextEditingController();
  final _fbsController = TextEditingController();
  final _restecgController = TextEditingController();
  final _thalachController = TextEditingController();
  final _exangController = TextEditingController();
  final _oldpeakController = TextEditingController();
  final _slopeController = TextEditingController();
  final _caController = TextEditingController();
  final _thalController = TextEditingController();
  String? _predictionResult;
  bool _isLoading = false;
  Map<String, dynamic>? _additionalData;

  Future<void> _submitData() async {
    if (_formKey.currentState?.validate() ?? false) {
      setState(() {
        _isLoading = true;
      });

      final age = int.parse(_ageController.text);
      final sex = int.parse(_sexController.text);
      final cp = int.parse(_cpController.text);
      final trestbps = int.parse(_trestbpsController.text);
      final chol = int.parse(_cholController.text);
      final fbs = int.parse(_fbsController.text);
      final restecg = int.parse(_restecgController.text);
      final thalach = int.parse(_thalachController.text);
      final exang = int.parse(_exangController.text);
      final oldpeak = double.parse(_oldpeakController.text);
      final slope = int.parse(_slopeController.text);
      final ca = int.parse(_caController.text);
      final thal = int.parse(_thalController.text);

      try {
        final response = await http.post(
          Uri.parse('http://127.0.0.1:5000/predict'), // Replace with your API endpoint
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal,
          }),
        );

        if (response.statusCode == 200) {
          final responseData = jsonDecode(response.body);
          setState(() {
            _predictionResult = responseData['prediction'];
          });

          // Fetch additional data and navigate to details screen
          await _fetchAndNavigateToDetails(responseData);
        } else {
          setState(() {
            _predictionResult = 'Error: Unable to get prediction';
          });
        }
      } catch (error) {
        setState(() {
          _predictionResult = 'Error: $error';
        });
      } finally {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _fetchAndNavigateToDetails(Map<String, dynamic> responseData) async {
    final modelAccuracyResponse = await http.get(Uri.parse('http://127.0.0.1:5000/plot_model_accuracy'));
    final targetDistributionResponse = await http.get(Uri.parse('http://127.0.0.1:5000/plot_target_distribution'));
    final dataDistributionResponse = await http.get(Uri.parse('http://127.0.0.1:5000/plot_data_distribution'));
    final chartResponse = await http.get(Uri.parse('http://127.0.0.1:5000/plot_chart'));

    final modelAccuracyPlot = modelAccuracyResponse.statusCode == 200 ? modelAccuracyResponse.bodyBytes : null;
    final targetDistributionPlot = targetDistributionResponse.statusCode == 200 ? targetDistributionResponse.bodyBytes : null;
    final dataDistributionPlot = dataDistributionResponse.statusCode == 200 ? dataDistributionResponse.bodyBytes : null;
    final chartPlot = chartResponse.statusCode == 200 ? chartResponse.bodyBytes : null;

    setState(() {
      _additionalData = {
        'modelAccuracyPlot': modelAccuracyPlot,
        'targetDistributionPlot': targetDistributionPlot,
        'dataDistributionPlot': dataDistributionPlot,
        'chartPlot': chartPlot,
      };
    });
  }

  void _resetForm() {
    _formKey.currentState?.reset();
    setState(() {
      _ageController.clear();
      _sexController.clear();
      _cpController.clear();
      _trestbpsController.clear();
      _cholController.clear();
      _fbsController.clear();
      _restecgController.clear();
      _thalachController.clear();
      _exangController.clear();
      _oldpeakController.clear();
      _slopeController.clear();
      _caController.clear();
      _thalController.clear();
      _predictionResult = null;
      _additionalData = null;
    });
  }

  void _showHelpSheet() {
    showModalBottomSheet(
      context: context,
      builder: (context) => HelpSheet(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Heart Disease Prediction'),
        backgroundColor: Colors.lightBlueAccent,
        elevation: 0,
        titleTextStyle: TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.bold,
          fontSize: 20,
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
              Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Input fields
                    Row(
                      children: [
                        Expanded(child: _buildTextFormField(_ageController, 'Age', 'Enter your age')),
                        SizedBox(width: 16),
                        Expanded(child: _buildTextFormField(_sexController, 'Sex', 'Enter your sex')),
                      ],
                    ),
                    SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(child: _buildTextFormField(_cpController, 'Chest Pain Type', 'Enter chest pain type')),
                        SizedBox(width: 16),
                        Expanded(child: _buildTextFormField(_trestbpsController, 'Resting Blood Pressure', 'Enter resting blood pressure (mm Hg)')),
                      ],
                    ),
                    SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(child: _buildTextFormField(_cholController, 'Cholesterol Level', 'Enter cholesterol level (mg/dl)')),
                        SizedBox(width: 16),
                        Expanded(child: _buildTextFormField(_fbsController, 'Fasting Blood Sugar', 'Enter 1 if fasting blood sugar > 120 mg/dl, otherwise 0')),
                      ],
                    ),
                    SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(child: _buildTextFormField(_restecgController, 'Resting Electrocardiographic Result', 'Enter resting electrocardiographic result')),
                        SizedBox(width: 16),
                        Expanded(child: _buildTextFormField(_thalachController, 'Maximum Heart Rate Achieved', 'Enter maximum heart rate')),
                      ],
                    ),
                    SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(child: _buildTextFormField(_exangController, 'Exercise Induced Angina', 'Enter 1 if exercise induced angina, otherwise 0')),
                        SizedBox(width: 16),
                        Expanded(child: _buildTextFormField(_oldpeakController, 'Oldpeak', 'Enter ST depression induced by exercise')),
                      ],
                    ),
                    SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(child: _buildTextFormField(_slopeController, 'Slope of ST Segment', 'Enter the slope of ST segment')),
                        SizedBox(width: 16),
                        Expanded(child: _buildTextFormField(_caController, 'Number of Major Vessels Colored', 'Enter number of major vessels colored by fluoroscopy')),
                      ],
                    ),
                    SizedBox(height: 16),
                    _buildTextFormField(_thalController, 'Thalassemia', 'Enter thalassemia value'),
                    SizedBox(height: 20),
                    if (_isLoading)
                      Center(child: CircularProgressIndicator(backgroundColor: Colors.lightBlueAccent,color: Colors.redAccent,)),
                    if (!_isLoading)
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Expanded(
                            child: ElevatedButton(
                              onPressed: _submitData,
                              child: Text('Predict',style: TextStyle(fontWeight: FontWeight.bold,color: Colors.white),),
                              style: ButtonStyle(
                                backgroundColor: MaterialStateProperty.all(Colors.lightBlueAccent),
                              ),
                            ),
                          ),
                          SizedBox(width: 10),
                          Expanded(
                            child: ElevatedButton(
                              onPressed: _resetForm,
                              child: Text('Reset',style: TextStyle(fontWeight: FontWeight.bold,color: Colors.white)),
                              style: ButtonStyle(
                                backgroundColor: MaterialStateProperty.all(Colors.redAccent),
                              ),
                            ),
                          ),
                          SizedBox(width: 10),
                          Expanded(
                            child: ElevatedButton(
                              onPressed: _showHelpSheet,
                              child: Text('Help',style: TextStyle(fontWeight: FontWeight.bold,color: Colors.white)),
                              style: ButtonStyle(
                                backgroundColor: MaterialStateProperty.all(Colors.orangeAccent),
                              ),
                            ),
                          ),
                        ],
                      ),
                    SizedBox(height: 10),
                    if (_predictionResult != null && !_isLoading)
                      ElevatedButton(
                        onPressed: () {
                          if (_additionalData != null) {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => DetailsScreen(
                                  predictionResult: _predictionResult!,
                                  modelAccuracyPlot: _additionalData!['modelAccuracyPlot'],
                                  targetDistributionPlot: _additionalData!['targetDistributionPlot'],
                                  dataDistributionPlot: _additionalData!['dataDistributionPlot'],
                                  chartPlot: _additionalData!['chartPlot'],
                                ),
                              ),
                            );
                          }
                        },
                        child: Text('View Details',style: TextStyle(fontWeight: FontWeight.bold,color: Colors.white)),
                        style: ButtonStyle(
                          backgroundColor: MaterialStateProperty.all(Colors.lightBlueAccent),
                        ),
                      ),
                  ],
                ),
              ),
              if (_predictionResult != null && !_isLoading)
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(
                    'Prediction: $_predictionResult',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTextFormField(TextEditingController controller, String label, String hint) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: TextFormField(
        controller: controller,
        keyboardType: TextInputType.number,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: OutlineInputBorder(),
        ),
        validator: (value) {
          if (value == null || value.isEmpty) {
            return 'Please enter some information';
          }
          return null;
        },
      ),
    );
  }
}
