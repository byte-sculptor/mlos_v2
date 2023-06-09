// -----------------------------------------------------------------------
// <copyright file="MlosContext.cs" company="Microsoft Corporation">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root
// for license information.
// </copyright>
// -----------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

using MlosProxyInternal = Proxy.Mlos.Core.Internal;

namespace Mlos.Core
{
    /// <summary>
    /// Creates an instance of MlosContext class.
    /// </summary>
    public static class MlosContextFactory
    {
        public static MlosContext Create()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return CreateUsingNamedMemoryMap();
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return CreateUsingAnonymousMemoryMap();
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        private static MlosContext CreateUsingAnonymousMemoryMap()
        {
            MlosContext mlosContext = Linux.AnonymousMemoryMlosContext.Create();
            return mlosContext;
        }

        private static MlosContext CreateUsingNamedMemoryMap()
        {
            return InterProcessMlosContext.Create();
        }
    }

    /// <summary>
    /// MlosContext encapsulates the shared memory regions for config and
    /// feedback for the Mlos.Agent when processing messages from smart
    /// components using their handlers.  It also includes a reference to the
    /// optimizer connection for those message handlers to use.
    /// </summary>
    /// <remarks>
    /// See Also: Mlos.Core/MlosContext.h for the corresponding C++ smart component side.
    /// </remarks>
    public abstract class MlosContext : IDisposable
    {
        /// <summary>
        /// Gets or sets the global Mlos context.
        /// </summary>
        /// <remarks>
        /// #TODO, we need an experiment class to be able pass the instance of the context.
        /// </remarks>
        public static MlosContext Instance { get; set; }

        #region Shared public objects

        /// <summary>
        /// Gets or sets the control channel instance.
        /// </summary>
        public ISharedChannel ControlChannel { get; protected set; }

        /// <summary>
        /// Gets or sets the feedback channel instance.
        /// </summary>
        public ISharedChannel FeedbackChannel { get; protected set; }

        /// <summary>
        /// Gets or sets the shared config manager.
        /// </summary>
        public SharedConfigManager SharedConfigManager { get; protected set; }

        /// <summary>
        /// Gets or sets the connection to the optimizer.
        /// </summary>
        /// <remarks>
        /// Typically this will be assigned for a deployment specific situation
        /// (see Mlos.Agent.Server/MlosAgentServer.cs for an example) prior to
        /// starting the Mlos.Agent and made available for message handlers to
        /// use. See SmartCacheExperimentSession.cs for an example.
        /// </remarks>
        public IOptimizerFactory OptimizerFactory { get; set; }

        #endregion

        /// <summary>
        /// Global memory region.
        /// </summary>
        protected SharedMemoryRegionView<MlosProxyInternal.GlobalMemoryRegion> globalMemoryRegionView;

        /// <summary>
        /// Control channel shared memory map.
        /// </summary>
        protected SharedMemoryMapView controlChannelMemoryMapView;

        /// <summary>
        /// Feedback channel shared memory map.
        /// </summary>
        protected SharedMemoryMapView feedbackChannelMemoryMapView;

        /// <summary>
        /// Notification event for the control channel.
        /// </summary>
        protected NamedEvent controlChannelNamedEvent;

        /// <summary>
        /// Notification event for the feedback channel.
        /// </summary>
        protected NamedEvent feedbackChannelNamedEvent;

        /// <summary>
        /// Indicates whether the object has been disposed.
        /// </summary>
        protected bool isDisposed;

        /// <summary>
        /// Protected implementation of Dispose pattern.
        /// </summary>
        /// <param name="disposing"></param>
        protected virtual void Dispose(bool disposing)
        {
            if (isDisposed || !disposing)
            {
                return;
            }

            // Close shared memory.
            //
            globalMemoryRegionView?.Dispose();
            globalMemoryRegionView = null;

            controlChannelMemoryMapView?.Dispose();
            controlChannelMemoryMapView = null;

            feedbackChannelMemoryMapView?.Dispose();
            feedbackChannelMemoryMapView = null;

            controlChannelNamedEvent?.Dispose();
            controlChannelNamedEvent = null;

            feedbackChannelNamedEvent?.Dispose();
            feedbackChannelNamedEvent = null;

            // Finally dispose the shared config manager.
            //
            SharedConfigManager?.Dispose();
            SharedConfigManager = null;

            isDisposed = true;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Terminate the control channel.
        /// </summary>
        public void TerminateControlChannel()
        {
            // Terminate the channel to avoid deadlocks if the buffer is full, and there is no active reader thread.
            //
            ControlChannel.SyncObject.TerminateChannel.Store(true);
            controlChannelNamedEvent.Signal();
        }

        /// <summary>
        /// Terminates the feedback channel.
        /// </summary>
        public void TerminateFeedbackChannel()
        {
            FeedbackChannel.SyncObject.TerminateChannel.Store(true);
            feedbackChannelNamedEvent.Signal();
        }

        /// <summary>
        /// Checks if the control channel is still active.
        /// </summary>
        /// <returns></returns>
        public bool IsControlChannelActive()
        {
            return !ControlChannel.SyncObject.TerminateChannel.Load();
        }

        /// <summary>
        /// Checks if the feedback channel is still active.
        /// </summary>
        /// <returns></returns>
        public bool IsFeedbackChannelActive()
        {
            return !FeedbackChannel.SyncObject.TerminateChannel.Load();
        }

        /// <summary>
        /// Gets a global memory region proxy object.
        /// </summary>
        public MlosProxyInternal.GlobalMemoryRegion GlobalMemoryRegion => globalMemoryRegionView.MemoryRegion();
    }
}
