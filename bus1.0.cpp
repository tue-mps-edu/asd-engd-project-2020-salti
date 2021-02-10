#include "bus.h"

Bus:: Bus(string license, string location, string manufacturer,
	int year, int passengers, Wheel* wheels, int nrWheels,
	Door* busDoor, int nrDoors, Fueltank* busTank) :Vehicle(license, location, manufacturer, year)
{
	cout << "Bus Constructor" << endl;
	this->passengers = passengers;
	this->wheels = wheels;
	this->nrWheels = nrWheels;
	this->busDoor = busDoor;
	this->nrDoors = nrDoors;
	this->busTank = busTank;
	this->busengine = Engine();
}

bool Bus::move(double distance)
{
	cout << "Bus moves" << distance << "kms" << endl;
	return true;
}
bool Bus::turn(double direction)
{
	cout << "Bus turns" << direction << endl;
	return true;
}
bool Bus::stop()
{
	cout << "Bus stopped" << endl;
	return true;
}
void Bus::accelerate()
{
	cout << "Bus accelerates" << endl;
}
void Bus::brake()
{
	cout << "Bus brakes" << endl;
}

void Bus:: loadPassengers(int passengers)
{
	this->passengers = this->passengers+passengers;
	cout << "New number of passengers are " << this->passengers << endl;
}

void Bus::unloadPassengers(int passengers)
{
	assert(this->passengers >= passengers);
	this->passengers = this->passengers - passengers;
	cout << "New number of passengers are " << this->passengers << endl;

}

ostream& operator<<(ostream& os, const Bus& c)
{
	os << "Bus :  " << endl;
	os << "		license:" << c.license << endl << "		location:" << c.location << endl
		<< "		manufacturer:" << c.manufacturer << endl << "		year:" << c.year << endl;
	os << "		nrofDoors:" << c.nrDoors << endl;
	os << "		nrofWheels:" << c.nrWheels << endl;
	os << "		nrOf Passengers:" << c.passengers << endl;

	for (int i = 0; i < c.nrWheels; i++) {

		os << *(c.wheels + i);
	}

	return os;
}

