import 'package:flutter/material.dart';
import 'dart:typed_data';

class DetailsScreen extends StatelessWidget {
  final String predictionResult;
  final Uint8List? modelAccuracyPlot;
  final Uint8List? targetDistributionPlot;
  final Uint8List? dataDistributionPlot;
  final Uint8List? chartPlot;

  const DetailsScreen({
    super.key,
    required this.predictionResult,
    this.modelAccuracyPlot,
    this.targetDistributionPlot,
    this.dataDistributionPlot,
    this.chartPlot,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Detailed Forecast: $predictionResult',style: TextStyle(color: Colors.white,fontWeight: FontWeight.bold),),
        backgroundColor: Colors.lightBlueAccent,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
              if (modelAccuracyPlot != null)
                _buildImagePlot(modelAccuracyPlot!),
              if (targetDistributionPlot != null)
                _buildImagePlot(targetDistributionPlot!),
              if (dataDistributionPlot != null)
                _buildImagePlot(dataDistributionPlot!),
              if (chartPlot != null)
                _buildImagePlot(chartPlot!),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImagePlot(Uint8List plotBytes) {
    return Column(
      children: [
        Image.memory(plotBytes),
        SizedBox(height: 20),
      ],
    );
  }
}
