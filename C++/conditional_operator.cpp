#include<iostream>

int main() {
  int x;
  std::cout << "Please enter a number between 0 and 100: " << '\n';
  std::cin >> x;
  (x > 100) ? (x = 100) : (x);
  (x < 0) ? (x = 0) : (x);
  std::cout << "Your number is " << x << '\n';
}
