using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using System;

namespace MLNetWorkInProgress
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(0);

            //New Way Start

            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                                        {
                                                        new TextLoader.Column("InsuranceCode", DataKind.Text, 0),
                                                        new TextLoader.Column("CarrierName", DataKind.Text, 1),
                                                        new TextLoader.Column("Address1", DataKind.Text, 2),
                                                        new TextLoader.Column("Address2", DataKind.Text, 3),
                                                        new TextLoader.Column("Zip", DataKind.Text, 4),
                                                        new TextLoader.Column("DefaultProfileType", DataKind.Text, 5),
                                                        new TextLoader.Column("CarrierId", DataKind.Text, 6),
                                                        new TextLoader.Column("State", DataKind.Text, 7),
                                                        new TextLoader.Column("Label", DataKind.R8, 8),
                                                    }
            });

            var data = textLoader.Read(@"data.csv");

            // Step 2: Pipeline


            var transformPipeline = mlContext.Transforms.Categorical.OneHotEncoding("State")
                         .Append(mlContext.Transforms.Categorical.OneHotEncoding("DefaultProfileType"))
                         .Append(mlContext.Transforms.Categorical.OneHotEncoding("InsuranceCode"))
                         .Append(mlContext.Transforms.Categorical.OneHotEncoding("Zip"))
                         //.Append(mlContext.Transforms.Text.FeaturizeText("CarrierName",
                         //                                                "CarrierName",
                         //                                                a =>
                         //                                                {
                         //                                                    a.KeepDiacritics = false;
                         //                                                    a.KeepPunctuations = false;
                         //                                                    a.TextCase =
                         //                                                        TextNormalizingEstimator
                         //                                                            .CaseNormalizationMode
                         //                                                            .Lower;
                         //                                                    a.OutputTokens = true;
                         //                                                    a.VectorNormalizer =
                         //                                                        TextFeaturizingEstimator
                         //                                                            .TextNormKind.L2;
                         //                                                }))
                         //.Append(mlContext.Transforms.Concatenate("Address",
                         //                                         "Address1",
                         //                                         "Address2"))
                         //.Append(mlContext.Transforms.Text.FeaturizeText("Address",
                         //                                                "Address",
                         //                                                a =>
                         //                                                {
                         //                                                    a.KeepDiacritics = false;
                         //                                                    a.KeepPunctuations = false;
                         //                                                    a.TextCase =
                         //                                                        TextNormalizingEstimator
                         //                                                            .CaseNormalizationMode
                         //                                                            .Lower;
                         //                                                    a.OutputTokens = true;
                         //                                                    a.VectorNormalizer =
                         //                                                        TextFeaturizingEstimator
                         //                                                            .TextNormKind.L2;
                         //                                                }))
                         .Append(mlContext.Transforms.Concatenate("Features",
                                                                  //"CarrierName",
                                                                  //"Address",
                                                                  "Zip",
                                                                  "State",
                                                                  "DefaultProfileType",
                                                                  "InsuranceCode"));

            var learner = mlContext.Regression.Trainers.StochasticDualCoordinateAscent(
                        labelColumn: DefaultColumnNames.Label, featureColumn: DefaultColumnNames.Features);




            var transformedData = transformPipeline.Fit(data).Transform(data);


            var model = learner.Fit(transformedData);

            var multiClassificationCtx = new MulticlassClassificationContext(mlContext);
            var metrics = multiClassificationCtx.Evaluate(transformedData, "Label");

            PrintClassificationMetrics("XRef", metrics);
        }

        private static void PrintClassificationMetrics(string name, MultiClassClassifierEvaluator.Result metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name}          ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Accuracy Macro: {metrics.AccuracyMacro}");
            Console.WriteLine($"*       Accuracy Micro: {metrics.AccuracyMicro}");
            Console.WriteLine($"*       Log Loss: {metrics.LogLoss}");
            Console.WriteLine($"*       Log Loss Reduction: {metrics.LogLossReduction}");
            Console.WriteLine($"*       Per Class Log Loss: {metrics.PerClassLogLoss}");
            Console.WriteLine($"*************************************************");
        }


    }
}
