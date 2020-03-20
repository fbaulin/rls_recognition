import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd
import logging
import sys


class Track:
    """ Класс для работы с траекториями

    Attributes:
         points(TrackPoint):    массив с точками траектории.
         obj_id(str):           идентификатор объекта.
         filename(str):         имя исходного файла траетории.

    """
    def __init__(self, filename=None):
        self.track_id = None        # идент. трека TODO: добавить запись трека
        self.points = None          # массив точек
        self.obj_id = None          # идентификатор объекта
        self.filename = None        # имя файла трека
        self.n_points = None        # число точек
        if filename is not None:
            if type(filename) == str:
                self.load_data(filename)
            else:
                raise TypeError(f'Требуется имя файла, но передан объект {type(filename)}')

    def load_data(self, filename: str, verbose=False):
        """Обработать файл траектории данные из файла

        Выполняет обработку выбранного файла данных (filename), и записывает (добавляет) данные в объект

        Args:
            filename(str): имя файла траектории.
            verbose(bool): флаг текстового вывода статуса исполнения комманды.

        """
        if verbose:
            print(f'Track file: {filename}')
        else:
            logging.info(f'Track file: {filename}')
        sep_idx = filename.rfind('\\')  # загрузка файла
        self.filename = filename[sep_idx+1:] if sep_idx != -1 else filename
        with open(filename) as trackfile:
            content = trackfile.read()
        self.points = []                        # создать пустой массив
        open_bracket_pos = content.find('{')    # найти первую открыв. скобку
        while open_bracket_pos != -1:           # пока не закончатся открывающие скобки продолжать обработку:
            open_bracket_pos += 1
            track_point_pos = content.index('"TrackPoint"', open_bracket_pos)   # найти фразу TrackPoint
            close_bracket_pos = content.find('}', track_point_pos)-1            # найти закр. скобку
            self.points.append(
                TrackPoint(content[open_bracket_pos:close_bracket_pos]))        # получить точку
            open_bracket_pos = content.find('{', close_bracket_pos+2)           # найти следующую '{' после '}'
        self.obj_id = self.points[0].obj_id         # взять id объекта из первого зондирования
        self.track_id = self.points[0].track_id     # идентификатор траектории
        self.n_points = len(self.points)            # число точек
        for point_obj in self.points:               # проверить остальные точки
            if point_obj.obj_id != self.obj_id:
                print(f'Иденитификатор точки #{point_obj.point_id} в {filename} не соответствуе идентификатору объекта')
            if point_obj.track_id != self.track_id:
                print(f'Иденитификатор точки #{point_obj.point_id} в {filename} не соответствуе идентификатору трека')

    def visualize(self):
        """Визуализация трека.

        Отладочная функция для просмотра изменения с принимаемого сигнала втечение регистрации трека.
        """
        # track_values = [np.array(p.get_values_for_polarization(complex_component='Amp')) for p in self.points]
        track_values = []
        for p in self.points:
            track_values_hor = np.array(p.get_values_for_polarization(
                polar='HighHor', complex_component='Re'))
            track_values_ver = np.array(p.get_values_for_polarization(
                polar='HighHor', complex_component='Im'))
            track_values.append(np.vstack((track_values_hor, track_values_ver)))
            print(f"r={p.obj_range}, t {p.pos_time[-5:-1]}")
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            # r, theta = np.meshgrid(np.arange(0, self.points[i].n_ranges*2, 1), self.points[i].azimuths)
            # ax.contourf(theta, r, track_values[i].transpose())
            # i=22
            single_data = track_values[i].transpose()
            single_data_h = single_data[:, 1]
            single_data_v = single_data[:, 4]

            # single_data = np.vstack((single_dataH, single_dataV))
            ax.plot(self.points[i].azimuths, single_data_h, self.points[i].azimuths, single_data_v)
            ax.set_ylim((-100, 100))
            ax.set_xlim((116.5, 121.5))
            # блок формирования преобразования Фурье
            # x = np.absolute(np.fft.rfft(track_values[i].transpose(), axis=0))
            # ax.contourf(x)
            # ax.set_ylim((0, 20))
            ax.set_title(f'{self.obj_id}: {i}')

        interval = 500  # мс
        ani = animation.FuncAnimation(fig, animate, self.n_points, interval=interval, blit=False)
        plt.show()
        del ani

    def output_image_fvs(self, transform='afft', n_features=16, n_means=None):
        """ Вывести отдельные вектора признаков для каждой точки зондирования.

            В цикле по точкам вывести их фрагменты, применив преобразование.

            Args:
                transform(str): тип преобразования.
                n_features(int): размерность вектора признаков.
                n_means(int): усредняемых реализаций.

            Returns:
                кортеж, содержащий векторы признаков и цели.
                (векторы признаков, цели),
                где вектры признаков - матрица
                    цели - вектора

        """
        x = np.zeros([self.n_points, n_features])
        t = [self.obj_id]*self.n_points
        for i_point in range(self.n_points):
            if transform == 'afft':
                x[i_point] = self.points[i_point].get_feature_set(n_features=n_features, mode=transform)
            elif transform == 'doppler':
                x[i_point] = self.points[i_point].get_doppler_fvs(n_features=n_features)

        if n_means is not None:

            n_datapoints = self.n_points//n_means
            if n_datapoints == 0:  # если траект. короче числа усредняемых реализаций
                n_datapoints = 1
                n_means = self.n_points
            mean_x = np.zeros([n_datapoints, n_features])
            for i in range(0, n_datapoints):
                # mean_x[i] = np.sum(x[i*3:i*3+n_means, :], axis=0)
                mean_x[i] = np.sum(x[i*n_means:(i+1)*n_means, :], axis=0)
            t = [self.obj_id]*n_datapoints
            # print(f'Trj #{self.track_id}:\t mean_x{mean_x.shape}, t({len(t)})')
            return mean_x, t

        return x, t

    @staticmethod
    def load_from_path(path='.\\dataset\\'):
        filelist = glob(path + '*.dat', recursive=False)  # get path
        dset = []
        for filename in filelist:
            try:
                track = Track(filename=filename)
            except FileExistsError:
                print("Проблемы при загрузке файла данных: файл не найден")
                continue
            except ValueError:
                print("Проблемы при загрузке файла данных: содержимое не опознано", )
                continue
            else:
                dset.append(track)
        return dset

    @staticmethod
    def set_to_csv(dset, filename="dataset.csv", n_features=16, n_means=None):
        """ Запись базы данных в csv.
        Для заданного набора данных сформировать векторы признаков и записать их в виде базы векторов признаков.

        Args:
            dset(list):         массив треков (объект класса Track).
            filename(str):      файл в который осуществляется вывод (нужно указывать с расширением).
            n_features(int):    размерность векторов признаков.
            n_means(int):       число реализаций (TrackPoint объектов) по которым выполняется усреднение.

        Returns:
            Файл filename.csv, содеражащий размеченную базу векторов признаков.

        """
        x_agg = []
        t_agg = []
        header = list(
            map(lambda spn: f'h_{spn}', range(n_features//2))) + list(
            map(lambda spn: f'v_{spn}', range(n_features//2)))
        for track in dset:
            x, t = track.output_image_fvs(n_features=n_features, n_means=n_means)
            x_agg += [x]
            t_agg += t
        # x = np.vstack(x_agg)
        # pd.DataFrame(x).to_csv(filename, header=header)
        pd_dset = pd.DataFrame(np.vstack(x_agg), columns=header)
        pd_dset['t'] = t_agg
        pd_dset.to_csv(filename, index=False)
        print('Dataset '+filename+' saved')


class TrackPoint:
    """ Класс точки маршрута

    Attributes:
        point_id(int):          идент. номер точки.
        n_azimuths(int):        число лучей по азимуту.
        azimuths(list):         список азимутов.
        n_ranges(int):          число элементов разрешения по дальности.
        obj_id(str):            идентификатор объекта.
        values(list):           значения сигнала в каналах - список[даль]{поляр:[действ., мним.]}.
        obj_range(float):       оценка расстояния до объекта.
        obj_azimuth(float):     оценка азимута объекта.
        track_id(int):          идентификационный номер траектории.

    """
    def __init__(self, data: str):
        self.point_id = None    # идентификатор точки
        self.n_azimuths = None  # число азимутальных секторов
        self.azimuths = None    # азимуты
        self.n_ranges = None    # число эл-тов разрешения по дальности
        self.obj_id = None      # идентификатор объекта
        self.values = None      # данные с приемных устройств список[даль]{поляр:[действ., мним.]}
        self.obj_range = None   # расстояние до объекта
        self.obj_azimuth = None  # азимут объекта
        self.track_id = None    # идентификационный номер траектории
        self.pos_time = None    # время
        self.polar_channels = ['HighHor', 'HighVer']     # каналы поляризации
        self.period_us = None   # интервал до следующего импульса
        if data is not None:
            self._process_data(data)    #

    # обработать строковую информацию
    def _process_data(self, trackpoint_text):
        polar_channels = self.polar_channels        # каналы поляризации
        self._extract_point_data(trackpoint_text)   # прочитать данные точки
        self.values = []
        for i_range in range(self.n_ranges):        # проход по эл-там дальности
            # вытащить блок для элемента дальности
            range_data_text = self._extract_subunit(trackpoint_text, 'Rng_' + str(i_range + 1))
            self.values.append(dict())
            for polar in polar_channels:            # проход по каналам поляризации
                # выделить данные канала поляризации
                polar_data_text = self._extract_subunit(range_data_text, polar)
                im_data = self._extract_num_array(polar_data_text, '"SrcIm"', val_type=int)     # прочитать мнимый канал
                re_data = self._extract_num_array(polar_data_text, '"SrcRe"', val_type=int)     # прочитать действит. к.
                self.values[i_range][polar] = [re_data, im_data]   # записать данные в

    # заполнить метаданные о точке
    def _extract_point_data(self, trackpoint_text):
        """ Парсинг текста с данными тракторного отсчета

        На вход передается сегмент файла, содержащего данные одной точки из полученных данных извлекается
        информация и записывается в качестве аттрибутов объекта.

        Args:
            trackpoint_text(str): фрагмент текстового файла, соответствующий одной пачке

        Returns:
            Результат парсинга заносится в текущий объект
        """
        self.azimuths = self._extract_num_array(trackpoint_text, '"AzimuthMass_deg"')   # извлеч азимут ант.
        self.n_azimuths = len(self.azimuths)
        self.n_ranges = self._extract_value(trackpoint_text, '"NumRangesPack"')         # число дальн. каналов
        self.point_id = self._extract_value(trackpoint_text, '"TrackPoint"')
        # get object type
        self.pos_time = self._extract_value(trackpoint_text, '"DateTimeFile"', val_type=str)
        start_index = trackpoint_text.index('"TypeCeilVOI":') + len('"TypeCeilVOI":') + 1
        start_index = trackpoint_text.index('"', start_index) + 1
        end_index = trackpoint_text.index('"', start_index)
        self.obj_id = trackpoint_text[start_index:end_index]
        self.obj_azimuth = self._extract_value(trackpoint_text, '"POI_Az_deg"', val_type=float)
        self.obj_range = self._extract_value(trackpoint_text, '"POI_Range_m"', val_type=float)
        self.track_id = self._extract_value(trackpoint_text, '"TrackNumber"')
        self.period_us = self._extract_num_array(trackpoint_text, '"Tper_usec"')

    # получить значения для заданной поляризации
    def get_values_for_polarization(self, polar='HighHor', complex_component=None):
        """ Сформировать массив значений, полученных с приемного устройства

        Args:
            polar(str):           тип поляризации - HighHor или HighVert
            complex_component:    компоненты None - обе, Re - действ, Im - мнимые, Amp - амплитуда (энергия)

        Returns:
            [list] -- 0 - расстояние, 1 - азимуты, 2 - действ. и мнимая компоненты (если complex_component)

        """
        values = []
        for i_range in range(self.n_ranges):
            values.append([[], ]*self.n_azimuths)
            for i_az in range(self.n_azimuths):
                if complex_component is None:       # записать [действ., мним.] компоненты
                    values[i_range][i_az] = [self.values[i_range][polar][0][i_az],
                                             self.values[i_range][polar][1][i_az]]
                elif complex_component == 'Re':     # записать действ. компоненту
                    values[i_range][i_az] = self.values[i_range][polar][0][i_az]
                elif complex_component == 'Im':     # записать мнимую компоненту
                    values[i_range][i_az] = self.values[i_range][polar][1][i_az]
                elif complex_component == 'Amp':    # рассчитать амплитудное значение
                    values[i_range][i_az] = self.values[i_range][polar][0][i_az]**2 + \
                                            self.values[i_range][polar][1][i_az]**2
                else:
                    raise AttributeError(
                        f'Арг. complex_component должен иметь значение Re, Im или Amp, задано {complex_component}')
        return values

    # получить список признаков
    def get_feature_set(self, mode='afft', n_features=16):
        """ Вывести строку признаков для одной отметки (объекта TrackPoint)

        Args:
            mode(str):          метод выделения признаков
            n_features(int):    число отсчетов

        Returns:
            вектор признаков

        """
        # собрать m матриц амплитуд для m каналов
        amplitudes = dict()
        az_shift = dict()
        for current_pol in self.polar_channels:             # для каждого канала поляризации рассчитать амплитуду
            loc_data = np.array(
                self.get_values_for_polarization(polar=current_pol, complex_component='Amp'))   # вытащить ампл. знач.
            # ra_shift = np.argmax(
            #     np.sum(loc_data, axis=1))                   # суммирование амплитуд по азимуту и выбор дальности
            # loc_data = loc_data[ra_shift, :]                # выделить информацию дальностного канала
            loc_data = np.sum(loc_data, axis=0)             #
            az_shift[current_pol] = np.argmax(loc_data)     # определение максимума
            amplitudes[current_pol] = loc_data              # сохранить амплитудные составл. компл. отсчетов
            if n_features > len(amplitudes[current_pol]):
                print('Число признаков превышает число отсчетов в совокупных спектрах')

        if mode == 'afft':  # AFFT амплитудных компонент комплексных отсчетов по временной (=азимутальной) оси
            hor_sp = np.abs(np.fft.rfft(amplitudes[self.polar_channels[0]]))    # form ampl. spectrum
            hor_sp = hor_sp[:n_features//2]         # take half of spectrum (Hermitian symmetry)
            hor_sp = hor_sp/np.mean(hor_sp)         # norm by mean value
            ver_sp = np.abs(np.fft.rfft(amplitudes[self.polar_channels[1]]))
            ver_sp = ver_sp[:n_features//2]
            ver_sp = ver_sp/np.mean(ver_sp)
            return np.hstack((hor_sp, ver_sp))      # cat vert & horz pol spectrums in a single feature vector
        else:
            TypeError("Неподдерживаемый метод формирования векторов признаков")

    # вывести доплеровский спектр сигнала
    def get_doppler_fvs(self, n_features=16):
        """ Вывести доплеровский спектр сигнала
        Args:
            n_features:          размерность вектора признаков

        Returns:
            doppler_spectrum    доплеровский спектр сигнала

        """
        # Последовательность действий:
        # вытащить значения для каждой поляризации, далее для каждой поляризации
        # выделить дальностный канал с наибольшей энергией
        # выполнить неэквидистантное преоборазование Фурье комплексных отсчетов
        #
        complex_counts = dict()                             # словарь с отсчетами по значениям поляризаций
        t_axis = np.array(np.cumsum(self.period_us[0:-1]))  # создать вектор с осью времени
        # вывести данные для поляризации
        for current_pol in self.polar_channels:  # для каждого канала рассчитать амплитуду
            # получить комплексные отсчеты для текущей поляризации
            loc_data =\
                np.array(self.get_values_for_polarization(polar=current_pol, complex_component='Re')) +\
                np.array(self.get_values_for_polarization(polar=current_pol, complex_component='Im')) * 1j
            # выделить дальностный канал с наибольшей энергией/контрастностью
            top_range_channel = np.argmax(np.sum(
                self.get_values_for_polarization(polar=current_pol, complex_component='Amp'), axis=1))
            complex_counts[current_pol] = loc_data[top_range_channel, :]
            # TODO: сделать суммирование результирующих спектров по нескольким каналам
            # TODO: добавить умножение на окно и совмещение окна с ДН антенны

        # в результате получить набор комплексных величин для каждого канала поляризации
        f_axis = np.linspace(0, 1e3, 10)        # сформировать вектор частот
        mti = np.exp(-1j*2*np.pi*t_axis.transpose()*f_axis)     # сформировать матрицу неэквидистантного преобразования
        # перемножить вектор комплексных отсчетов на матрицу

    def visualize(self):
        """ вывести изображение поля """
        # Using linspace so that the endpoint of 360 is included...
        # azimuths = np.radians(self.azimuths)
        azimuths = np.radians(self.azimuths)
        zeniths = np.arange(0, self.n_ranges, 1)
        r, theta = np.meshgrid(zeniths, azimuths)
        values = np.array(self.get_values_for_polarization(complex_component='Amp')).transpose()
        # -- Plot... ------------------------------------------------
        fig, ax = plt.subplots(subplot_kw=dict(projection=None))
        ax.contourf(theta, r, values)
        plt.show()
        return ax

    @staticmethod   # найти в тексте параметр и считать его значение
    def _extract_value(trackpoint_text, parameter_name, val_type=None):
        val_type = int if val_type is None else val_type
        start_index = trackpoint_text.index(parameter_name) + len(parameter_name) + 2
        end_index = trackpoint_text.index(',', start_index)
        return val_type(trackpoint_text[start_index:end_index])

    @staticmethod   # найти в тексте параметр и считать его массив его значений
    def _extract_num_array(pol_channel: str, parameter_name: str, val_type=float):
        parameter_index = pol_channel.index(parameter_name) + len(parameter_name) + 2
        open_index = pol_channel.index('[', parameter_index) + 1
        close_index = pol_channel.index(']', open_index)
        array_segment = pol_channel[open_index:close_index]
        array_segment = array_segment.replace('\n', '').strip().split(',')
        return list(map(val_type, array_segment))

    @staticmethod   # найти параметр и выделить блок, ограниченный {} скобками
    def _extract_subunit(text_unit: str, parameter_name: str):
        parameter_end_index = text_unit.index(parameter_name)
        start_idx = text_unit.find('{', parameter_end_index)
        stop_idx = text_unit.find('}', start_idx)
        while text_unit.count('{', start_idx, stop_idx) > text_unit.count('}', start_idx, stop_idx+1):
            stop_idx = text_unit.index('}', stop_idx+1)
        return text_unit[start_idx+1:stop_idx]  # extracts the data range data


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    dataset = Track.load_from_path(path='.\\dataset\\raw_data\\')
    # dataset[1].points[0].visualize()
    Track.set_to_csv(dataset, filename='.\\dataset\\kolomna_val.csv', n_features=32, n_means=7)
