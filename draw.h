#ifdef DRAW

#include <SFML/Graphics.hpp>

void Draw(sf::RenderWindow *window, float3 *positions, int n) {
    window->clear();

    for (int i = 0; i < n; i++) {
        sf::CircleShape point(1.0f);
        point.setPosition((float) (window->getSize().x * positions[i].x), (float) (window->getSize().y * (1 - positions[i].y)));
        point.setOrigin(point.getRadius(), point.getRadius());
        point.setFillColor(sf::Color::White);
        window->draw(point);
    }

    window->display();
    sf::Event event{};
    while (window->pollEvent(event)) {
        if (event.type == sf::Event::Closed)
            window->close();
    }
}

#endif

