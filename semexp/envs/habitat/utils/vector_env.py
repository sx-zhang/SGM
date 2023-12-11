#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import signal
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import gym

import habitat
import numpy as np
from gym.spaces import Dict as SpaceDict
from habitat.config import Config
from habitat.core.env import Env, Observations, RLEnv
from habitat.core.logging import logger
from habitat.core.utils import tile_images

try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
EPISODE_COMMAND = "current_episode"
CURRENT_EPISODE_COMMAND = "get_current_eps"
PLAN_ACT_AND_PREPROCESS = "plan_act_and_preprocess"
REACHABILITY_COMMAND = "get_reachability_map"
FRONTIER_COMMAND = "get_frontier_map"
FMM_DISTS_COMMAND = "get_fmm_dists"
COUNT_EPISODES_COMMAND = "count_episodes"
EPISODE_OVER = "episode_over"
GET_METRICS = "get_metrics"


def _make_env_fn(
    config: Config, dataset: Optional[habitat.Dataset] = None, rank: int = 0
) -> Env:
    """Constructor for default habitat `env.Env`.

    :param config: configuration for environment.
    :param dataset: dataset for environment.
    :param rank: rank for setting seed of environment
    :return: `env.Env` / `env.RLEnv` object
    """
    habitat_env = Env(config=config, dataset=dataset)
    habitat_env.seed(config.SEED + rank)
    return habitat_env


class VectorEnv:
    r"""Vectorized environment which creates multiple processes where each
    process runs its own environment. Main class for parallelization of
    training and evaluation.


    All the environments are synchronized on step and reset methods.
    """

    observation_spaces: List[SpaceDict]
    action_spaces: List[SpaceDict]
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ) -> None:
        """..

        :param make_env_fn: function which creates a single environment. An
            environment can be of type `env.Env` or `env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            `_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess  must be started before any other GPU useage.
        :param workers_ignore_signals: Whether or not workers will ignore
            SIGINT and SIGTERM and instead will only exit when :ref:`close` is called
        """
        self._is_waiting = False
        self._is_closed = True

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args,
            make_env_fn,
            workers_ignore_signals=workers_ignore_signals,
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))
        self.observation_spaces = [read_fn() for read_fn in self._connection_read_fns]
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [read_fn() for read_fn in self._connection_read_fns]
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]
        self._paused = []

    @property
    def num_envs(self):
        r"""number of individual environments."""
        return self._num_envs - len(self._paused)

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)

        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for habitat.RLEnv and habitat.Env
                    if isinstance(env, habitat.RLEnv) or isinstance(env, gym.Env):
                        # habitat.RLEnv
                        observations, reward, done, info = env.step(**data)
                        if auto_reset_done and done:
                            observations, info = env.reset()
                        connection_write_fn((observations, reward, done, info))
                    elif isinstance(env, habitat.Env):
                        # habitat.Env
                        observations = env.step(**data)
                        if auto_reset_done and env.episode_over:
                            observations = env.reset()
                        connection_write_fn(observations)
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    if isinstance(command, str):
                        connection_write_fn(getattr(env, command))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(env, function_name)()
                    else:
                        result = getattr(env, function_name)(**function_args)
                    connection_write_fn(result)

                # TODO: update CALL_COMMAND for getting attribute like this
                elif command == EPISODE_COMMAND:
                    connection_write_fn(env.current_episode)

                elif command == CURRENT_EPISODE_COMMAND:
                    connection_write_fn(env.current_eps)

                elif command == PLAN_ACT_AND_PREPROCESS:
                    observations, reward, done, info = env.plan_act_and_preprocess(data)
                    if auto_reset_done and done:
                        observations, info = env.reset()
                    connection_write_fn((observations, reward, done, info))

                elif command == REACHABILITY_COMMAND:
                    reachability_map, fmm_dist = env.get_reachability_map(data)
                    connection_write_fn((reachability_map, fmm_dist))

                elif command == FRONTIER_COMMAND:
                    frontier_map = env.get_frontier_map(data)
                    connection_write_fn(frontier_map)

                elif command == FMM_DISTS_COMMAND:
                    fmm_dists = env.get_fmm_dists(data)
                    connection_write_fn(fmm_dists)

                elif command == COUNT_EPISODES_COMMAND:
                    connection_write_fn(len(env.episodes))

                elif command == EPISODE_OVER:
                    connection_write_fn(env.episode_over)

                elif command == GET_METRICS:
                    result = env.get_metrics()
                    connection_write_fn(result)

                else:
                    raise NotImplementedError

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        )
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(
            worker_connections, parent_connections, env_fn_args
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    workers_ignore_signals,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    def current_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((EPISODE_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def get_current_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((CURRENT_EPISODE_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def count_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((COUNT_EPISODES_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def episode_over(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((EPISODE_OVER, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def get_metrics(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((GET_METRICS, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def reset(self):
        r"""Reset all the vectorized environments

        :return: list of outputs from the reset method of envs.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        obs, infos = zip(*results)

        self._is_waiting = False
        return np.stack(obs), infos

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.

        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: Dict[str, Any]):
        r"""Step in the index_env environment in the vector.

        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_async(self, data: List[Union[int, str, Dict[str, Any]]]) -> None:
        r"""Asynchronously step in the environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """
        # Backward compatibility
        if isinstance(data[0], (int, np.integer, str)):
            data = [{"action": {"action": action}} for action in data]

        self._is_waiting = True
        for write_fn, args in zip(self._connection_write_fns, data):
            write_fn((STEP_COMMAND, args))

    def step_wait(self) -> List[Observations]:
        r"""Wait until all the asynchronized environments have synchronized."""
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> List[Any]:
        r"""Perform actions in the vectorized environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.step_async(data)
        return self.step_wait()

    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env.

        :param index: which env to pause. All indexes after this one will be
            shifted down by one.

        This is useful for not needing to call steps on all environments when
        only some are active (for example during the last episodes of running
        eval episodes).
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        r"""Resumes any paused envs."""
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function (which is passed by name) on the selected env and
        returns the result.

        :param index: which env to call the function on.
        :param function_name: the name of the function to call on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).

        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
        """
        self._is_waiting = True
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(self._connection_write_fns, func_args):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def render(self, mode: str = "human", *args, **kwargs) -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image."""
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
        tile = tile_images(images)
        if mode == "human":
            from habitat.core.utils import try_cv2_import

            cv2 = try_cv2_import()

            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    def plan_act_and_preprocess(self, inputs):
        self._assert_not_closed()
        self._is_waiting = True
        for e, write_fn in enumerate(self._connection_write_fns):
            write_fn((PLAN_ACT_AND_PREPROCESS, inputs[e]))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        obs, rews, dones, infos = zip(*results)
        self._is_waiting = False
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_reachability_map(self, inputs):
        self._assert_not_closed()
        self._is_waiting = True
        for e, write_fn in enumerate(self._connection_write_fns):
            write_fn((REACHABILITY_COMMAND, inputs[e]))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        reachability_maps, fmm_dists = zip(*results)
        self._is_waiting = False
        return np.stack(reachability_maps), np.stack(fmm_dists)

    def get_frontier_map(self, inputs):
        self._assert_not_closed()
        self._is_waiting = True
        for e, write_fn in enumerate(self._connection_write_fns):
            write_fn((FRONTIER_COMMAND, inputs[e]))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        frontier_maps = results
        self._is_waiting = False
        return np.stack(frontier_maps)

    def get_fmm_dists(self, inputs):
        self._assert_not_closed()
        self._is_waiting = True
        for e, write_fn in enumerate(self._connection_write_fns):
            write_fn((FMM_DISTS_COMMAND, inputs[e]))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        fmm_dists = results
        self._is_waiting = False
        return np.stack(fmm_dists)

    def _assert_not_closed(self):
        assert (
            not self._is_closed
        ), "Trying to operate on a SubprocVecEnv after calling close()"

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ThreadedVectorEnv(VectorEnv):
    r"""Provides same functionality as `VectorEnv`, the only difference is it
    runs in a multi-thread setup inside a single process.

    `VectorEnv` runs in a multi-proc setup. This makes it much easier to debug
    when using `VectorEnv` because you can actually put break points in the
    environment methods. It should not be used for best performance.
    """

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Env] = _make_env_fn,
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_read_queues, parent_write_queues = zip(
            *[(Queue(), Queue()) for _ in range(self._num_envs)]
        )
        self._workers = []
        for parent_read_queue, parent_write_queue, env_args in zip(
            parent_read_queues, parent_write_queues, env_fn_args
        ):
            thread = Thread(
                target=self._worker_env,
                args=(
                    parent_write_queue.get,
                    parent_read_queue.put,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                ),
            )
            self._workers.append(thread)
            thread.daemon = True
            thread.start()
        return (
            [q.get for q in parent_read_queues],
            [q.put for q in parent_write_queues],
        )


class ListEnv:
    r"""Vectorized environment which creates multiple processes where each
    process runs its own environment. Main class for parallelization of
    training and evaluation.


    All the environments are synchronized on step and reset methods.
    """

    observation_spaces: List[SpaceDict]
    action_spaces: List[SpaceDict]
    _auto_reset_done: bool

    def __init__(
        self,
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
    ) -> None:
        """..

        :param make_env_fn: function which creates a single environment. An
            environment can be of type `env.Env` or `env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            `_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        """

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        self._auto_reset_done = auto_reset_done

        self.envs = [make_env_fn(*efargs) for efargs in env_fn_args]

        self.observation_spaces = [env.get_spaces()[0] for env in self.envs]
        self.action_spaces = [env.get_spaces()[1] for env in self.envs]
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]

    @property
    def num_envs(self):
        r"""number of individual environments."""
        return len(self.envs)

    def reset(self):
        r"""Reset all the vectorized environments

        :return: list of outputs from the reset method of envs.
        """
        results = []
        for env in self.envs:
            results.append(env.reset())
        obs, infos = zip(*results)

        return np.stack(obs), infos

    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> List[Any]:
        r"""Perform actions in the vectorized environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        if isinstance(data[0], (int, np.integer, str)):
            data = [{"action": {"action": action}} for action in data]
        results = []
        for env, action in zip(self.envs, data):
            results.append(env.step(**action))
        obs, rews, dones, infos = [list(x) for x in zip(*results)]
        for i in range(self.num_envs):
            done = dones[i]
            env = self.envs[i]
            if self._auto_reset_done and done:
                obs_i, infos_i = env.reset()
                obs[i] = obs_i
                infos[i] = infos_i
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def plan_act_and_preprocess(self, data):
        results = []
        for env, env_data in zip(self.envs, data):
            results.append(env.plan_act_and_preprocess(data))
        obs, rews, dones, infos = [list(x) for x in zip(*results)]
        for i in range(self.num_envs):
            done = dones[i]
            env = self.envs[i]
            if self._auto_reset_done and done:
                obs_i, infos_i = env.reset()
                obs[i] = obs_i
                infos[i] = infos_i
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
