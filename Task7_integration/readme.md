Методичка: http://www.apmath.spbu.ru/ru/staff/eremin/files/task7_2016.pdf

Во вложении тесты для ваших решений задачи вычисления определённого интеграла.

* `py/`
	* `test_integration.py` — набор тестов:
		* `test_nc_degree()` — проверит АСТ для ИКФ Ньютона — Котса
		* `test_gauss_degree()` — проверит АСТ для КФ типа Гаусса
		* `test_nc_convergence()` - проверит сходимость СКФ Ньютона — Котса при половинном делении шага
		* `test_gauss_convergence()` - проверит сходимость СКФ Гаусса при половинном делении шага
		* `test_optimal()` - проверит выбор оптимального шага для СКФ Ньютона — Котса

	* `integration.py` — тут должны быть ваши квадратурные формулы и вспомогательные функции:
		* `moments()` — расчёт моментов весовой функции (реализован для единичного веса)
		* `runge()` — оценка погрешности по правилу Рунге
		* `aitken()` — оценка скорости сходимости по правилу Эйткена
		* `equidist()` — выбор равномерно распределённых узлов на отрезке (уже реализована)
		* `quad()` — «малая» КФ Ньютона — Котса
		* `quad_gauss()` — «малая» ИКФ типа Гаусса
		* `composite_quad()` — составная КФ Ньютона — Котса 
		(уже реализована)
		* `composite_gauss()` — составная КФ Гаусса (уже реализована)
		* `doubling_nc()` — интегрирование СКФ НК удвоением числа шагов 
		до заданной точности с использованием теоретической скорости 
		сходимости в правиле Рунге
		* `doubling_nc_aitken()` — интегрирование СКФ НК удвоением числа 
		шагов до заданной точности с использованием правила Эйткена оценки
		скорости сходимости  
		* `doubling_gauss()` — интегрирование СКФ Гаусса удвоением 
		числа шагов до заданной точности с использованием 
		теоретической скорости сходимости в правиле Рунге
		* `doubling_gauss_aitken()` — интегрирование СКФ НК удвоением
		числа шагов до заданной точности с использованием правила 
		Эйткена оценки скорости сходимости
		* `optimal_nc()` — интегрирование СКФ НК с предложением 
		оптимального числа шагов по заданной точности с 
		использованием правила Эйткена оценки скорости сходимости
		
		
		
		
Рекомендации:
* Для расчёта узлов формулы Гаусса используйте функцию `numpy.roots(c)`,
 где `c` — массив коэффициентов полинома от старшего к младшему 
* Для вычисления моментов начните с нулевого и выразите все остальные 
 через предыдущие, как это сделано в методичке до второго порядка. 
 Напишите функцию, вычисляющую произвольное число моментов.
* Если в варианте ненулевая `beta`, а не `alpha`, то можно использовать 
 ту же самую формулу расчёта моментов, но заменив `x` на `t = -x`. Тогда
 в переменной `t` вес будет иметь особенность в левой точке, как если бы
 `alpha` была ненулевой, а `beta` — нулевой. Реализуйте возможность запустить
 тесты для любого значения глобальной переменной `variant`
* MAXITER — это глобальная переменная, которая нужна, чтобы вы не ушли
в бесконечный цикл. Если достигнуто предёльное число итераций, то на выходе
значения, заведомо не проходящие тест      
* При реализации решения с оптимальным шагом реализуйте проверку того,
что требуемая точность достигнута.
* Обратите внимание, что оценка скорости по Эйткену не должна использоваться,
если под логарифмом стоит отрицательное число 
(например, выдавайте в этом случае `-1`, то есть скорость сходимости,
которая заведомо не должна быть, ибо это — расходимость). Кроме того, 
оценкой Эйткена не следует пользоваться, если она сильно 
(например, более чем на 2) отличается от теоретической (АСТ малой формулы + 1).  
 
		
