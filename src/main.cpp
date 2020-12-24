#include "../nanoflann.hpp"
#include "../SO3.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort


# define M_PI 3.14159265358979323846


typedef Geometry::Vec3 Point;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatRowMajor;
typedef nanoflann::KDTreeEigenMatrixAdaptor<MatRowMajor> KDTree;


std::vector<size_t> sortIndicies(const std::vector<double>& v)
{
	//https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}


Mat HStack(const std::vector<Point>& points)
{
	size_t numCols = points.size();
	size_t numRows = 3;
	Mat hstackedMat(numRows, numCols);
	for (size_t colIndex = 0; colIndex < numCols; ++colIndex)
	{
		hstackedMat.col(colIndex) = points[colIndex];
	}
	return hstackedMat;
}


Mat ReadXYZFile(std::string filename)
{
	std::cout << "reading in: " << filename << "\n";

	std::ifstream pointsFile(filename);
	std::vector<Point> points;

	std::string line;
	double x, y, z;
	int nextSpace;
	const std::string delimiter = " ";

	while (!pointsFile.eof())
	{
		std::getline(pointsFile, line);

		nextSpace = line.find(delimiter);
		x = std::stod(line.substr(0, nextSpace).c_str());
		line = line.substr(nextSpace + 1);

		nextSpace = line.find(delimiter);
		y = std::stod(line.substr(0, nextSpace).c_str());
		line = line.substr(nextSpace + 1);

		nextSpace = line.find(delimiter);
		z = std::stod(line.substr(0, nextSpace).c_str());

		points.push_back(Point(x, y, z));
	}

	return HStack(points);
}


void WriteMatToXYZFile(const Mat& pointcloud, std::string filename)
{
	std::ofstream fileOut;
	fileOut.open(filename);
	for (int colIndex = 0; colIndex < pointcloud.cols(); ++colIndex)
	{
		fileOut << pointcloud.data()[3 * colIndex] << " "
			<< pointcloud.data()[3 * colIndex + 1] << " "
			<< pointcloud.data()[3 * colIndex + 2] << "\n";
	}
	fileOut.close();
	std::cout << filename << " written.\n";
}


Point GetCenterOfGravity(Mat& pointcloud)
{
	return pointcloud.rowwise().sum() / pointcloud.cols();
}


void FindMatches(const KDTree& targetPointsTree, const Mat& pointsToMove,
	std::vector<Eigen::Index>& matches, std::vector<double>& sqrDists)
{
	// finds the index of closest point in target_points (as kdTree) for every point in points_to_move
	const size_t numResults = 1;
	int printPeriod = 1000;
	for (int queryIndex = 0; queryIndex < pointsToMove.cols(); ++queryIndex)
	{
		if (queryIndex % printPeriod == 0)
		{
			std::cout << "Finding matches " << queryIndex << "/" << pointsToMove.cols() << "...\r";
		}
		std::vector<Eigen::Index> retIndicies(numResults);
		std::vector<double> outSqrDists(numResults);

		const double query_point[3] =
		{ 
			pointsToMove.data()[queryIndex * 3], 
			pointsToMove.data()[queryIndex * 3 + 1], 
			pointsToMove.data()[queryIndex * 3 + 2] 
		};

		targetPointsTree.index->knnSearch(&query_point[0], numResults, &retIndicies[0], &outSqrDists[0]);

		matches.push_back(retIndicies[0]);
		sqrDists.push_back(outSqrDists[0]);
	}
	std::cout << "Finding matches " << pointsToMove.cols() << "/" << pointsToMove.cols() << " done\n";
}


Mat CalculateOptimalRotation(const Mat& pointsToMove, const Mat& pointsTarget)
{
	Mat PQ = pointsToMove * pointsTarget.transpose();
	Eigen::JacobiSVD svd(PQ, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Mat R = svd.matrixV() * svd.matrixU().transpose();

	if (R.determinant() < 0)
	{
		Mat V = svd.matrixV();
		V.col(2) = -1 * V.col(2);
		R = V * svd.matrixU().transpose();
	}
	return R;
}


MatRowMajor GetRowMajorCopy(const Mat& mat)
{
	MatRowMajor asRowMajor(mat.cols(), mat.rows());
	for (int i = 0; i < mat.size(); ++i)
	{
		asRowMajor.data()[i] = mat.data()[i];
	}
	return asRowMajor;
}



void DemoICPWithTwoPointClouds(std::string filename1, std::string filename2)
{
	// read is two original pointclouds
	Mat pointsTarget = ReadXYZFile(filename1);
	Mat pointsToMove = ReadXYZFile(filename2);

	// we will search for matches in the target pointcloud, so we build a kd tree of it
	// for that we need a rowmajor matrix
	MatRowMajor TargetAsRowMajor = GetRowMajorCopy(pointsTarget);
	KDTree treeOfTarget(3, TargetAsRowMajor, 20);
	treeOfTarget.index->buildIndex();

	// run iterative closest point
	int numIterations = 100;
	double stoppingMSE = 0.005;
	for (int i = 0; i < numIterations; ++i)
	{
		// find corresponding pairs
		std::vector<Eigen::Index> matchIndiciesInTarget;
		std::vector<double> matchSqrDistances;
		FindMatches(treeOfTarget, pointsToMove, matchIndiciesInTarget, matchSqrDistances);
		double MSE = 0.0;
		for (int i = 0; i < matchSqrDistances.size(); MSE += matchSqrDistances[i++]);
		MSE /= (double)matchSqrDistances.size();
		std::cout << i + 1 << ": Mean squared error before transformation = " << MSE << "\n";
		if (MSE < stoppingMSE)
		{
			std::cout << "Stopping as goal MSE reached" << "\n";
			break;
		}

		// create a reordered matrix of the target points
		Mat targetReordered(pointsToMove.rows(), pointsToMove.cols());
		for (int i = 0; i < matchIndiciesInTarget.size(); ++i)
		{
			targetReordered.col(i) = pointsTarget.col(matchIndiciesInTarget[i]);
		}

		// center pointclouds
		Point centerOfGravityCurrentTarget = GetCenterOfGravity(targetReordered);
		Point centerOfGravityCurrentToMove = GetCenterOfGravity(pointsToMove);
		Mat targetCentered = targetReordered.colwise() - centerOfGravityCurrentTarget;
		Mat toMoveCentered = pointsToMove.colwise() - centerOfGravityCurrentToMove;

		// get optimal rotation for the centered pointcloud 
		Mat R = CalculateOptimalRotation(toMoveCentered, targetCentered);
		Point t = centerOfGravityCurrentTarget - R * centerOfGravityCurrentToMove;
		pointsToMove = (R * pointsToMove).colwise() + t;

		std::stringstream file_name;
		file_name << "output/ICPafter" << i + 1 << "interations.xyz";
		WriteMatToXYZFile(pointsToMove, file_name.str());
	}
}


void DemoTrICPWithPartiallyOverlappingPointClouds(std::string filename1, std::string filename2)
{
	// read is two original pointclouds
	Mat pointsTarget = ReadXYZFile(filename1);
	Mat pointsToMove = ReadXYZFile(filename2);

	// we will search for matches in the target pointcloud, so we build a kd tree of it
	// for that we need a rowmajor matrix
	MatRowMajor TargetAsRowMajor = GetRowMajorCopy(pointsTarget);
	KDTree treeOfTarget(3, TargetAsRowMajor, 20);
	treeOfTarget.index->buildIndex();

	// run iterative closest point
	int numIterations = 100;
	double stoppingMSE = 0.005;
	double ratioToKeep = 0.6;
	int matchesToKeep = (int)((double)pointsToMove.cols() * ratioToKeep);
	for (int i = 0; i < numIterations; ++i)
	{
		// find corresponding pairs
		std::vector<Eigen::Index> matchIndiciesInTarget;
		std::vector<double> matchSqrDistances;
		FindMatches(treeOfTarget, pointsToMove, matchIndiciesInTarget, matchSqrDistances);

		// create a set of relevant pairs
		std::vector<size_t> matchOrder = sortIndicies(matchSqrDistances);
		// current error
		double MSE = 0.0;
		for (int i = 0; i < matchesToKeep; MSE += matchSqrDistances[matchOrder[i++]]);
		MSE /= (double)matchesToKeep;
		std::cout << i + 1 << ": Mean squared error before transformation = " << MSE << "\n";
		if (MSE < stoppingMSE)
		{
			std::cout << "Stopping as goal MSE reached" << "\n";
			break;
		}

		// create trimmed pc to move, trimmed and reordered pc target
		Mat target(pointsTarget.rows(), matchesToKeep);
		Mat toMove(pointsToMove.rows(), matchesToKeep);
		for (int i = 0; i < matchesToKeep; ++i)
		{
			toMove.col(i) = pointsToMove.col(matchOrder[i]);
			target.col(i) = pointsTarget.col(matchIndiciesInTarget[matchOrder[i]]);
		}

		// center pointclouds
		Point centerOfGravityCurrentTarget = GetCenterOfGravity(target);
		Point centerOfGravityCurrentToMove = GetCenterOfGravity(toMove);
		Mat targetCentered = target.colwise() - centerOfGravityCurrentTarget;
		Mat toMoveCentered = toMove.colwise() - centerOfGravityCurrentToMove;

		// get optimal rotation for the centered pointcloud 
		Mat R = CalculateOptimalRotation(toMoveCentered, targetCentered);
		Point t = centerOfGravityCurrentTarget - R * centerOfGravityCurrentToMove;
		pointsToMove = (R * pointsToMove).colwise() + t;

		std::stringstream file_name;
		file_name << "output/TrICPafter" << i + 1 << "interations.xyz";
		WriteMatToXYZFile(pointsToMove, file_name.str());
	}

}


Mat GenerateSmallRandomRotation(double maxAngleX, double maxAngleY, double maxAngleZ)
{
	double radX = (((double)rand() / RAND_MAX) * 2.0 * maxAngleX - maxAngleX) / 180.0 * M_PI;
	double radY = (((double)rand() / RAND_MAX) * 2.0 * maxAngleY - maxAngleY) / 180.0 * M_PI;
	double radZ = (((double)rand() / RAND_MAX) * 2.0 * maxAngleZ - maxAngleZ) / 180.0 * M_PI;
	Geometry::Mat3 RX, RY, RZ;
	RX << 1.0, 0.0, 0.0,
		0.0, std::cos(radX), -1.0 * std::sin(radX),
		0.0, std::sin(radX), std::cos(radX);
	RY << std::cos(radY), 0.0, std::sin(radY),
		0.0, 1.0, 0.0,
		-1.0 * std::sin(radY), 0.0, std::cos(radY);
	RZ << std::cos(radZ), -1.0 * std::sin(radZ), 0.0,
		std::sin(radZ), std::cos(radZ), 0.0,
		0.0, 0.0, 1.0;
	return RZ * RY * RX;
}


Mat CreateOtherPointCloud(Mat& pointcloud,
	double relativeshift = 0.1, double relativeNoise = 0.01, double maxAngleX = 10.0, double maxAngleY = 10.0, double maxAngleZ = 10.0)
{
	Mat R = GenerateSmallRandomRotation(maxAngleX, maxAngleY, maxAngleZ);
	Point ranges = pointcloud.rowwise().maxCoeff() - pointcloud.rowwise().minCoeff();
	Point shift;
	shift << (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX;
	shift = shift.cwiseProduct(ranges * 2.0 * relativeshift) - ranges * relativeshift;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, ranges.mean() * relativeNoise);
	Mat otherPointCloud = (R * pointcloud).colwise() + shift;
	for (int i = 0; i < otherPointCloud.size(); ++i)
	{
		otherPointCloud.data()[i] += distribution(generator);
	}
	return otherPointCloud;
}


int main(int argc, char **argv)
{
	std::cout << "\nType in a number to run a program.\n";
	std::cout << "\t(1) - ICP with one pointcloud and a randomly shifted, rotated, noisy copy of it\n";
	std::cout << "\t(2) - ICP with two highly overlapping pointclouds\n";
	std::cout << "\t(3) - TrICP with two partially overlapping pointclouds\n";

	int choice;
	std::cin >> choice;

	switch (choice)
	{
	case 1:
	{
		DemoICPWithTwoPointClouds("input/fountain_a_centered.xyz", "input/fountain_a_centered_with_noise.xyz");
		break;
	}
	case 2:
	{
		DemoICPWithTwoPointClouds("input/walls1.xyz", "input/walls2.xyz");
		break;
	}
	case 3:
	{
		DemoTrICPWithPartiallyOverlappingPointClouds("input/fountain_a.xyz", "input/fountain_b_modified.xyz");
		break;
	}
	default:
	{
		std::cout << "\nThat was not a valid choice.\n\n";
		break;
	}
	}
	
	return 0;
}