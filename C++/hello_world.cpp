#include<iostream>

int main() {
  int x, y;
  std::cout << "Hello, world! It's me." << '\n';
  std::cout << "Please enter two numbers: " << '\n';
  std::cout << "First number:\t" << '\n';
  std::cin >> x;
  std::cout << "Second number:\t" << '\n';
  std::cin >> y;
  double a = (double)x / y;
  std::cout << "The sum of these numbers is:\t" << x + y << '\n';
  std::cout << "The fraction is:\t" << a << '\n';
}
