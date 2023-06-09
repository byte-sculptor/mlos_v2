// -----------------------------------------------------------------------
// <copyright file="SharedMemoryMapView.Linux.cs" company="Microsoft Corporation">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root
// for license information.
// </copyright>
// -----------------------------------------------------------------------

using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;

namespace Mlos.Core.Linux
{
    /// <summary>
    /// Linux implementation of shared memory map view.
    /// </summary>
    public sealed class SharedMemoryMapView : Mlos.Core.SharedMemoryMapView
    {
        /// <summary>
        /// Create a new shared memory view.
        /// </summary>
        /// <param name="sharedMemoryMapName"></param>
        /// <param name="sharedMemorySize"></param>
        /// <returns></returns>
        public static new SharedMemoryMapView CreateNew(string sharedMemoryMapName, ulong sharedMemorySize)
        {
            // Try to unlink existing shared memory.
            //
            _ = Native.SharedMemoryUnlink(sharedMemoryMapName);

            // Create shared memory view.
            //
            var sharedMemoryMapView = new SharedMemoryMapView(
                sharedMemoryMapName,
                sharedMemorySize,
                Native.OpenFlags.O_CREAT | Native.OpenFlags.O_RDWR | Native.OpenFlags.O_EXCL);

            return sharedMemoryMapView;
        }

        /// <summary>
        /// Creates or opens a shared memory view.
        /// </summary>
        /// <param name="sharedMemoryMapName"></param>
        /// <param name="sharedMemorySize"></param>
        /// <returns></returns>
        public static new SharedMemoryMapView CreateOrOpen(string sharedMemoryMapName, ulong sharedMemorySize)
        {
            // Create or open shared memory view.
            //
            var sharedMemoryMapView = new SharedMemoryMapView(
                sharedMemoryMapName,
                sharedMemorySize,
                Native.OpenFlags.O_CREAT | Native.OpenFlags.O_RDWR);

            return sharedMemoryMapView;
        }

        /// <summary>
        /// Opens an existing shared memory view.
        /// </summary>
        /// <param name="sharedMemoryMapName"></param>
        /// <param name="sharedMemorySize"></param>
        /// <returns></returns>
        public static new SharedMemoryMapView OpenExisting(string sharedMemoryMapName, ulong sharedMemorySize)
        {
            return new SharedMemoryMapView(
                sharedMemoryMapName,
                sharedMemorySize,
                Native.OpenFlags.O_RDWR);
        }

        /// <summary>
        /// Opens an anonymous shared memory map from the file descriptor.
        /// </summary>
        /// <param name="sharedMemoryFd"></param>
        /// <returns></returns>
        public static SharedMemoryMapView OpenFromFileDescriptor(IntPtr sharedMemoryFd)
        {
            int result = Native.FileStats(sharedMemoryFd, out FileStatus fileStatus);
            if (result != 0)
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }

            Console.WriteLine($"SharedMemoryMapView OpenFromFileDescriptor {sharedMemoryFd}, {fileStatus.Size}, {fileStatus.Uid}");

            return new SharedMemoryMapView(
                sharedMemoryFd,
                (ulong)fileStatus.Size);
        }

        private SharedMemoryMapView(IntPtr sharedMemoryFd, ulong sharedMemorySize)
        {
            sharedMemoryHandle = new SharedMemorySafeHandle(sharedMemoryFd);

            CreateMemoryMap(sharedMemorySize);
        }

        private SharedMemoryMapView(string sharedMemoryMapName, ulong sharedMemorySize, Native.OpenFlags openFlags)
        {
            this.sharedMemoryMapName = sharedMemoryMapName;

            // Create shared memory view.
            //
            sharedMemoryHandle = Native.SharedMemoryOpen(
                sharedMemoryMapName,
                openFlags,
                Native.ModeFlags.S_IRUSR | Native.ModeFlags.S_IWUSR);

            if (sharedMemoryHandle.IsInvalid)
            {
                int errno = Marshal.GetLastWin32Error();

                throw new FileNotFoundException(
                    $"Failed to shm_open {sharedMemoryMapName}",
                    innerException: new Win32Exception(errno));
            }

            CreateMemoryMap(sharedMemorySize);
        }

        private void CreateMemoryMap(ulong sharedMemorySize)
        {
            if (Native.FileTruncate(sharedMemoryHandle, (long)sharedMemorySize) == -1)
            {
                int errno = Marshal.GetLastWin32Error();
                throw new FileNotFoundException(
                    $"Failed to ftruncate {sharedMemoryMapName} {sharedMemoryHandle}",
                    innerException: new Win32Exception(errno));
            }

            Buffer = Native.MapMemory(
                address: IntPtr.Zero,
                length: sharedMemorySize,
                protFlags: Native.ProtFlags.PROT_READ | Native.ProtFlags.PROT_WRITE,
                mapFlags: Native.MapFlags.MAP_SHARED,
                handle: sharedMemoryHandle,
                offset: 0);
            if (Buffer == Native.InvalidPointer)
            {
                int errno = Marshal.GetLastWin32Error();
                throw new InvalidOperationException(
                    $"Failed to mmap {sharedMemoryMapName} {sharedMemoryHandle}",
                    innerException: new Win32Exception(errno));
            }

            MemSize = sharedMemorySize;
        }

        /// <summary>
        /// Protected implementation of Dispose pattern.
        /// </summary>
        /// <param name="disposing"></param>
        protected override void Dispose(bool disposing)
        {
            if (isDisposed || !disposing)
            {
                return;
            }

            // Close shared memory.
            //
            sharedMemoryHandle?.Dispose();

            if (CleanupOnClose)
            {
                // Unlink shared map. Ignore the errors.
                //
                if (sharedMemoryMapName != null)
                {
                    _ = Native.SharedMemoryUnlink(sharedMemoryMapName);
                }

                CleanupOnClose = false;
            }

            isDisposed = true;
        }

        private readonly SharedMemorySafeHandle sharedMemoryHandle;

        private readonly string sharedMemoryMapName;
    }
}
