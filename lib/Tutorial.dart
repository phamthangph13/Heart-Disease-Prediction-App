import 'package:flutter/material.dart';

class HelpSheet extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                'Data Entry Guide',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.redAccent),
              ),
              SizedBox(height: 16),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Age: Enter your age.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Sex: Enter 1 for male and 0 for female.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Chest Pain Type: Enter the value corresponding to the type of chest pain.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Resting Blood Pressure (mm Hg): Enter your resting blood pressure.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Cholesterol Level (mg/dl): Enter your blood cholesterol level.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Fasting Blood Sugar > 120 mg/dl: Enter 1 if fasting blood sugar is greater than 120 mg/dl, otherwise enter 0.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Resting Electrocardiographic Result: Enter the value corresponding to the ECG result.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Maximum Heart Rate Achieved: Enter the maximum heart rate achieved during exercise.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Exercise Induced Chest Pain: Enter 1 if you experience chest pain during exercise, otherwise enter 0.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'ST Depression Induced by Exercise vs Rest: Enter the ST depression value.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Slope of ST Segment Peak Exercise: Enter the slope value of the ST segment.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Number of Major Vessels Colored by Fluoroscopy: Enter the number of vessels.'),
              _buildHelpItem(Icons.arrow_forward_outlined, 'Thalassemia: Enter the thalassemia value.'),
              SizedBox(height: 20),
              Center(
                child: ElevatedButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: Text(
                    'Close',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(Colors.lightBlueAccent),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHelpItem(IconData icon, String text) {
    return Row(
      children: [
        Icon(icon, color: Colors.redAccent, size: 20),
        SizedBox(width: 8),
        Expanded(
          child: Text(
            text,
            style: TextStyle(fontSize: 18),
          ),
        ),
      ],
    );
  }
}
